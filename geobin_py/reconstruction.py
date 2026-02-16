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
        self.activation = activation
        self.parent = None
        self.children = []
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
            self.W = []
            self.b = []
            
            if "model" in g:
                g_model = g["model"]
                w_keys = sorted([k for k in g_model.keys() if k.startswith("W_")], 
                                key=lambda x: int(x.split('_')[1]))
                for wk in w_keys:
                    bk = wk.replace("W_", "b_")
                    W_mat = g_model[wk][:]
                    b_vec = g_model[bk][:]
                    
                    # FIX: Always transpose weights from Julia (Col-Major) -> Python (Row-Major)
                    # We must rely on the source behavior, not the shape heuristic.
                    W_mat = W_mat.T
                    
                    self.W.append(W_mat)
                    self.b.append(b_vec)
            else:
                # Legacy fallback
                keys = list(g.keys())
                w_keys = sorted([k for k in keys if "weight" in k],
                                key=lambda x: int(re.search(r"l(\d+)\.", x).group(1)) if re.search(r"l(\d+)\.", x) else 999)
                for wk in w_keys:
                    bk = wk.replace("weight", "bias")
                    self.W.append(g[wk][:])
                    self.b.append(g[bk][:])

            if self.W:
                self.L = len(self.W)
                self.input_dim = self.W[0].shape[1]
            
            # --- 2. Load Topology ---
            if "parent_ids" not in g:
                return

            parent_ids = g["parent_ids"][:]
            layer_idxs = g["layer_idxs"][:]
            # Centroids are (N, D) in the fixed Julia script
            centroids  = g["centroids"][:] 
            qlw_flat   = g["qlw_flat"][:]
            qlw_offsets = g["qlw_offsets"][:]
            
            num_nodes = len(parent_ids)

            # Safety check for orientation (just in case)
            if centroids.shape[0] != num_nodes and centroids.shape[1] == num_nodes:
                centroids = centroids.T

            if len(centroids) < num_nodes:
                print(f"    ⚠️ Data mismatch! parent_ids={num_nodes}, centroids={len(centroids)}")
                num_nodes = len(centroids)

            # --- 3. Build Nodes ---
            nodes = []
            for i in range(num_nodes):
                if i+1 < len(qlw_offsets):
                    act = qlw_flat[qlw_offsets[i]:qlw_offsets[i+1]]
                else:
                    act = np.array([], dtype=int)

                vol = g["volumes"][i] if "volumes" in g and i < len(g["volumes"]) else -1
                bnd = bool(g["bounded"][i]) if "bounded" in g and i < len(g["bounded"]) else False
                
                node = Region(i, layer_idxs[i], vol, centroids[i], act, bnd)
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
            
            if self.W:
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
            
            # Geometry Update: Z = W(Ax+c) + b
            # Note: W_mat is already correctly oriented (Out, In) due to the fix above
            W_hat = W_mat @ parent.Alw
            b_hat = W_mat @ parent.clw + b_vec
            
            for child in parent.children:
                q = child.activation
                if len(q) != W_mat.shape[0]:
                    continue

                s_vec = -2.0 * q + 1.0 
                
                child.Dlw = s_vec[:, None] * W_hat
                child.glw = -(s_vec * b_hat)
                
                child.Alw = q[:, None] * W_hat
                child.clw = q * b_hat
                
                queue.append(child)

if __name__ == "__main__":
    pass