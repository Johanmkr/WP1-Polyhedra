import h5py
import numpy as np
import time
import re

class Region:
    def __init__(self, idx, layer, volume, centroid, activation, is_bounded):
        self.id = idx
        self.layer = layer
        self.volume = volume
        self.is_bounded = is_bounded
        self.centroid = centroid
        self.activation = activation  # q vector
        
        # Topology
        self.parent = None
        self.children = []
        
        # Geometry
        self.Alw = None 
        self.clw = None
        self.Dlw = None 
        self.glw = None 

    def get_path_inequalities(self):
        if self.Dlw is None:
            return None, None
        
        D_list, g_list = [], []
        node = self
        while node is not None:
            if node.Dlw is not None and node.Dlw.shape[0] > 0:
                D_list.append(node.Dlw)
                g_list.append(node.glw)
            node = node.parent
            
        if not D_list:
            dim = self.Alw.shape[1] if self.Alw is not None else 2
            return np.zeros((0, dim)), np.zeros(0)

        return np.vstack(D_list[::-1]), np.concatenate(g_list[::-1])

    

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
            
            # Sort l1, l2...
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
            
            # --- 2. Load Topology ---
            if "parent_ids" not in g:
                raise ValueError("Tree topology (parent_ids) missing.")

            parent_ids = g["parent_ids"][:]
            layer_idxs = g["layer_idxs"][:]
            centroids  = g["centroids"][:] 
            qlw_flat   = g["qlw_flat"][:]
            qlw_offsets = g["qlw_offsets"][:]
            
            # Safety Check for Array Sizes
            num_nodes = len(parent_ids)
            if len(centroids) < num_nodes:
                print(f"    ⚠️ WARNING: Data mismatch! parent_ids={num_nodes}, centroids={len(centroids)}")
                print("    Truncating tree construction to match available data.")
                num_nodes = len(centroids)

            # --- 3. Build Nodes ---
            nodes = []
            for i in range(num_nodes):
                # Safe slicing for qlw
                if i+1 < len(qlw_offsets):
                    start, end = qlw_offsets[i], qlw_offsets[i+1]
                    act = qlw_flat[start:end]
                else:
                    act = np.array([], dtype=int)

                # Safe loading of attributes
                vol = g["volumes"][i] if i < len(g["volumes"]) else -1
                bnd = bool(g["bounded"][i]) if i < len(g["bounded"]) else False
                
                node = Region(
                    idx=i,
                    layer=layer_idxs[i],
                    volume=vol,
                    centroid=centroids[i],
                    activation=act,
                    is_bounded=bnd
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
            self._hydrate_geometry()

    def _hydrate_geometry(self):
        if not self.root: return

        self.root.Alw = np.eye(self.input_dim)
        self.root.clw = np.zeros(self.input_dim)
        self.root.Dlw = np.zeros((0, self.input_dim))
        self.root.glw = np.zeros(0)
        
        queue = [self.root]
        
        while queue:
            parent = queue.pop(0)
            
            if not parent.children: continue
            if parent.layer >= len(self.W): continue
                
            W_mat = self.W[parent.layer]
            b_vec = self.b[parent.layer]
            
            # Z = W(Ax+c) + b
            W_hat = W_mat @ parent.Alw
            b_hat = W_mat @ parent.clw + b_vec
            
            for child in parent.children:
                q = child.activation
                if len(q) != W_mat.shape[0]:
                    # Activation size mismatch (geometry likely invalid/incomplete)
                    continue

                s_vec = -2.0 * q + 1.0 
                
                child.Dlw = s_vec[:, None] * W_hat
                child.glw = -(s_vec * b_hat)
                
                child.Alw = q[:, None] * W_hat
                child.clw = q * b_hat
                
                queue.append(child)
                
    def assign_points_to_regions(self, points: np.ndarray, target_layer: int) -> np.ndarray:
        """
        Given an array of points (N, input_dim), returns an array of shape (N,)
        containing the Region ID that each point belongs to at the target_layer.
        Returns -1 for points that fall into a region not present in the tree.
        """
        N = points.shape[0]
        region_assignments = np.full(N, -1, dtype=int)
        
        if self.root is None:
            return region_assignments
            
        # Queue stores tuples of: (current_node, indices_of_points_in_this_node)
        queue = [(self.root, np.arange(N))]
        
        while queue:
            node, pt_idx = queue.pop(0)
            
            if len(pt_idx) == 0:
                continue
                
            # If we've reached the desired layer, record the region ID for these points
            if node.layer == target_layer:
                region_assignments[pt_idx] = node.id
                continue
                
            # If we are at a leaf but haven't reached the target layer, we can't route further
            if not node.children:
                continue
                
            # To route points to the next layer, we compute their pre-activations (Z)
            # using the affine transformation mapping (Alw, clw) from the root to this node.
            W_mat = self.W[node.layer]
            b_vec = self.b[node.layer]
            
            W_hat = W_mat @ node.Alw
            b_hat = W_mat @ node.clw + b_vec
            
            # Z = X @ W_hat^T + b_hat
            Z = points[pt_idx] @ W_hat.T + b_hat
            
            # Compute activation signature for these points
            Q = (Z > 0).astype(int) # shape: (len(pt_idx), num_neurons)
            
            # Route points to the matching child region
            for child in node.children:
                # Find points whose activation signature matches this child's signature
                match_mask = np.all(Q == child.activation, axis=1)
                child_pt_idx = pt_idx[match_mask]
                
                if len(child_pt_idx) > 0:
                    queue.append((child, child_pt_idx))
                    
        return region_assignments

if __name__ == "__main__":
    pass