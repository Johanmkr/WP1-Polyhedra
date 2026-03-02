import h5py
import numpy as np
import re
from collections import defaultdict

class Region:
    def __init__(self, idx, layer, volume_ex, volume_es, centroid, activation, is_bounded, active_indices=None):
        self.id = idx
        self.layer = layer
        self.volume_ex = volume_ex
        self.volume_es = volume_es
        self.is_bounded = is_bounded
        self.centroid = centroid
        self.activation = activation  # q vector (1D numpy array)
        self.active_indices = active_indices if active_indices is not None else []
        
        # Topology
        self.parent = None
        self.children = []

    def get_activation_path(self):
        """Traces up the tree to get the sequence of activations from Layer 1 to this region."""
        path = []
        curr = self
        while curr.parent is not None and curr.layer > 0:
            path.append(curr.activation)
            curr = curr.parent
        return path[::-1]

    def __repr__(self):
        return f"<Region ID:{self.id} L:{self.layer}>"


class Tree:
    def __init__(self, h5_file: str, epoch: int):
        self.h5_file = h5_file
        self.epoch = epoch
        self.W = []
        self.b = []
        self.root = None
        self.input_dim = 0
        self.L = 0
        self.leaves = []
        
        self._load_and_construct()

    def get_regions_at_layer(self, layer: int):
        regions = []
        if self.root is None: return regions
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.layer == layer:
                regions.append(node)
            elif node.layer < layer:
                queue.extend(node.children)
        return regions

    def get_path_inequalities(self, region: Region, active_only: bool = False):
        """
        Dynamically recalculates the path inequalities D*x <= g for a given region.
        Consumes virtually zero permanent memory.
        """
        q_path = region.get_activation_path()
        
        # Handle the root node (unbounded, no constraints)
        if not q_path:
            return np.zeros((0, self.input_dim)), np.zeros(0)
            
        A_curr = np.eye(self.input_dim)
        c_curr = np.zeros(self.input_dim)
        
        D_list = []
        g_list = []
        
        for l, q in enumerate(q_path):
            W = self.W[l]
            b = self.b[l]
            
            W_hat = W @ A_curr
            b_hat = W @ c_curr + b
            
            # s_vec is -1 if q=0, and 1 if q=1
            s_vec = -2.0 * q + 1.0
            
            D_local = s_vec[:, None] * W_hat
            g_local = -(s_vec * b_hat)
            
            D_list.append(D_local)
            g_list.append(g_local)
            
            A_curr = q[:, None] * W_hat
            c_curr = q * b_hat
            
        D_full = np.vstack(D_list)
        g_full = np.concatenate(g_list)
        
        if active_only and len(region.active_indices) > 0:
            idx = np.array(region.active_indices)
            return D_full[idx], g_full[idx]
            
        return D_full, g_full

    def _load_and_construct(self):
        print(f"  - Loading Epoch {self.epoch}...")
        
        with h5py.File(self.h5_file, "r") as f:
            if "epochs" not in f:
                 raise KeyError("No 'epochs' group in file.")
            
            epoch_key = f"epoch_{self.epoch}"
            if epoch_key not in f["epochs"]:
                raise KeyError(f"Epoch {self.epoch} not found.")
                
            g = f["epochs"][epoch_key]
            
            # --- 1. Load Weights ---
            keys = list(g.keys())
            weight_keys = [k for k in keys if "weight" in k]
            
            def get_layer_idx(k):
                m = re.search(r"l(\d+)\.", k)
                return int(m.group(1)) if m else 999
            weight_keys.sort(key=get_layer_idx)
            
            if not weight_keys:
                raise ValueError("No weights found in HDF5.")

            self.W = []
            self.b = []
            
            for w_key in weight_keys:
                b_key = w_key.replace("weight", "bias")
                self.W.append(g[w_key][:])
                self.b.append(g[b_key][:])

            self.L = len(self.W)
            self.input_dim = self.W[0].shape[1]
            
            # --- 2. Load Topology & Attributes ---
            if "parent_ids" not in g:
                raise ValueError("Tree topology (parent_ids) missing.")

            parent_ids = g["parent_ids"][:]
            layer_idxs = g["layer_idxs"][:]
            centroids  = g["centroids"][:] 
            
            # Load Activation Signatures
            qlw_flat = g["qlw_flat"][:]
            qlw_offsets = g["qlw_offsets"][:]
            
            # Load Active Indices (if they exist from an Exact Tree)
            has_active = "active_flat" in g
            if has_active:
                active_flat = g["active_flat"][:]
                active_offsets = g["active_offsets"][:]
            
            num_nodes = len(parent_ids)

            # --- 3. Build Lightweight Nodes ---
            nodes = []
            for i in range(num_nodes):
                # Slice the flattened qlw array
                if i+1 < len(qlw_offsets):
                    start, end = qlw_offsets[i], qlw_offsets[i+1]
                    act = qlw_flat[start:end]
                else:
                    act = np.array([], dtype=int)
                    
                # Slice the flattened active indices array
                active_idx = []
                if has_active and i+1 < len(active_offsets):
                    start, end = active_offsets[i], active_offsets[i+1]
                    active_idx = active_flat[start:end]

                vol_ex = g["volumes_ex"][i] if i < len(g["volumes_ex"]) else -1
                vol_es = g["volumes_es"][i] if i < len(g["volumes_es"]) else -1
                bnd = bool(g["bounded"][i]) if i < len(g["bounded"]) else False
                
                node = Region(
                    idx=i,
                    layer=layer_idxs[i],
                    volume_ex=vol_ex,
                    volume_es=vol_es,
                    centroid=centroids[i],
                    activation=act,
                    is_bounded=bnd,
                    active_indices=active_idx
                )
                nodes.append(node)
            
            # --- 4. Link ---
            self.root = None
            for i, p_id in enumerate(parent_ids[:num_nodes]):
                if p_id == -1:
                    self.root = nodes[i]
                elif 0 <= p_id < num_nodes:
                    parent = nodes[p_id]
                    child = nodes[i]
                    child.parent = parent
                    parent.children.append(child)
            
            self.leaves = [n for n in nodes if not n.children]

    def perform_number_count(self, data, y=None) -> dict:
        """
        Records the number of points per class that fall into each region at every layer.
        Optimized by running a standard forward pass first.
        """
        def to_numpy(tensor_or_array):
            if hasattr(tensor_or_array, "detach"):
                return tensor_or_array.detach().cpu().numpy()
            return np.asarray(tensor_or_array)

        if y is not None:
            batches = [(to_numpy(data), to_numpy(y).flatten())]
        else:
            def batch_generator():
                for X_batch, y_batch in data:
                    yield to_numpy(X_batch), to_numpy(y_batch).flatten()
            batches = batch_generator()

        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        if self.root is None:
            return dict(counts)
            
        for X_np, y_np in batches:
            N = X_np.shape[0]
            
            # 1. Perform a highly optimized forward pass to get all activations
            Q_path = []
            A = X_np
            for l in range(self.L):
                Z = A @ self.W[l].T + self.b[l]
                Q = (Z > 0).astype(int)
                Q_path.append(Q)
                A = Q * Z
                
            # 2. Route points through the tree
            # Queue stores tuples of: (current_node, indices_of_points_in_this_node)
            queue = [(self.root, np.arange(N))]
            
            while queue:
                node, pt_idx = queue.pop(0)
                
                if len(pt_idx) == 0:
                    continue
                    
                # A. Update the class counts for the current region/layer
                classes, class_counts = np.unique(y_np[pt_idx], return_counts=True)
                for cls, count in zip(classes, class_counts):
                    counts[node.layer][node.id][cls] += int(count)
                    
                # B. If we are at a leaf, we can't route points any deeper
                if not node.children:
                    continue
                    
                # C. Grab the pre-computed activations for the NEXT layer
                next_layer_Q = Q_path[node.layer]
                
                # D. Route points to matching children
                for child in node.children:
                    # Find points whose activation signature matches this child
                    match_mask = np.all(next_layer_Q[pt_idx] == child.activation, axis=1)
                    child_pt_idx = pt_idx[match_mask]
                    
                    if len(child_pt_idx) > 0:
                        queue.append((child, child_pt_idx))
                        
        # Clean up the defaultdicts into standard Python dictionaries
        result = {}
        for layer, regions in counts.items():
            result[layer] = {}
            for reg_id, class_counts in regions.items():
                result[layer][reg_id] = dict(class_counts)
                
        return result

if __name__ == "__main__":
    pass