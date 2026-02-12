import h5py
import numpy as np
import time

class Region:
    """
    Represents a polyhedral region in the tree.
    Attributes match the Julia structure: Dlw (inequalities), Alw (affine map), etc.
    """
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
        
        # Geometry (Computed during tree initialization)
        self.Alw = None 
        self.clw = None
        self.Dlw = None 
        self.glw = None 

    def get_children(self):
        return self.children

    def get_path_inequalities(self):
        """
        Traverses ancestors to return the full matrix of constraints (A_full * x <= b_full).
        """
        if self.Dlw is None:
            raise ValueError("Geometry not hydrated. Tree not fully initialized.")

        D_list = []
        g_list = []
        node = self
        
        while node is not None:
            D_list.append(node.Dlw)
            g_list.append(node.glw)
            node = node.parent
            
        # Stack Top-Down (Root -> Leaf)
        # Note: Root constraints are typically empty for unbounded input
        return np.vstack(D_list[::-1]), np.concatenate(g_list[::-1])

    def __repr__(self):
        vol_str = "Inf" if self.volume == float('inf') else f"{self.volume:.4f}"
        return (f"<Region L{self.layer} ID:{self.id} Vol:{vol_str} "
                f"Bounded:{self.is_bounded} Children:{len(self.children)}>")


class Tree:
    def __init__(self, h5_file: str, epoch: int):
        """
        Initializes the Tree by reading structure and weights from an HDF5 file.
        
        Args:
            h5_file: Path to the .h5 file created by the Julia script.
            epoch: The specific epoch integer to load (e.g., 10).
        """
        self.h5_file = h5_file
        self.epoch = epoch
        
        # Initialize attributes
        self.W = []
        self.b = []
        self.root = None
        self.input_dim = 0
        self.L = 0
        
        # Perform loading and hydration
        self._load_and_construct()

    def get_regions_at_layer(self, layer: int):
        """Returns a list of all Region objects at a specific depth."""
        regions = []
        queue = [self.root]
        
        while queue:
            current_region = queue.pop(0)
            
            if current_region.layer == layer:
                regions.append(current_region)
            elif current_region.layer < layer:
                queue.extend(current_region.children)
        return regions

    def _load_and_construct(self):
        print(f"Loading Tree (Epoch {self.epoch}) from {self.h5_file}...")
        t0 = time.time()
        
        with h5py.File(self.h5_file, "r") as f:
            key = f"epoch_{self.epoch}"
            if key not in f:
                raise KeyError(f"Epoch {self.epoch} not found in {self.h5_file}")
                
            g = f[key]
            
            # --- 1. Load Model Hyperplanes (Weights & Biases) ---
            g_model = g["model"]
            self.L = g["depth"][()]
            self.input_dim = g["input_dim"][()]
            
            self.W = []
            self.b = []
            for l in range(1, self.L + 1):
                # Transpose W because HDF5 reads as (In, Out) but we need (Out, In)
                self.W.append(g_model[f"W_{l}"][:].T)
                self.b.append(g_model[f"b_{l}"][:])
            
            # --- 2. Load Topology & Attributes ---
            parent_ids = g["parent_ids"][:]
            layer_idxs = g["layer_idxs"][:]
            volumes    = g["volumes"][:]
            bounded    = g["bounded"][:] 
            
            # Transpose centroids (N, Dim) -> (Dim, N) -> Transpose back
            centroids  = g["centroids"][:].T 
            
            qlw_flat    = g["qlw_flat"][:]
            qlw_offsets = g["qlw_offsets"][:]
            
            num_nodes = len(parent_ids)
            
            # --- 3. Reconstruct Region Objects ---
            nodes = []
            for i in range(num_nodes):
                start, end = qlw_offsets[i], qlw_offsets[i+1]
                act = qlw_flat[start:end]
                
                node = Region(
                    idx=i,
                    layer=layer_idxs[i],
                    volume=volumes[i],
                    centroid=centroids[i],
                    activation=act,
                    is_bounded=bool(bounded[i])
                )
                nodes.append(node)
            
            # --- 4. Link Parents & Children ---
            self.root = None
            for i, p_id in enumerate(parent_ids):
                if p_id == -1:
                    self.root = nodes[i]
                else:
                    parent = nodes[p_id]
                    child = nodes[i]
                    child.parent = parent
                    parent.children.append(child)
            
            # --- 5. Hydrate Geometry (Compute D, A, b matrices) ---
            self._hydrate_geometry()
            
            dt = time.time() - t0
            print(f"Tree constructed in {dt:.2f}s. Loaded {num_nodes} regions.")

    def _hydrate_geometry(self):
        """
        Recursively calculates the geometric matrices (Dlw, glw, Alw, clw) 
        for every node in the tree using the loaded weights.
        """
        # Root Geometry: Unbounded identity map
        self.root.Alw = np.eye(self.input_dim)
        self.root.clw = np.zeros(self.input_dim)
        self.root.Dlw = np.zeros((0, self.input_dim)) # Empty constraints
        self.root.glw = np.zeros(0)
        
        queue = [self.root]
        
        while queue:
            parent = queue.pop(0)
            
            if not parent.children:
                continue
                
            # Parent (Layer L) -> Children (Layer L+1) uses W[L]
            # Since Python lists are 0-indexed, W[0] corresponds to Layer 1 transition
            if parent.layer >= len(self.W):
                continue
                
            W_mat = self.W[parent.layer]
            b_vec = self.b[parent.layer]
            
            # Pre-calculate Parent's Affine Output: z = W(Ax+c) + b
            W_hat = W_mat @ parent.Alw
            b_hat = W_mat @ parent.clw + b_vec
            
            for child in parent.children:
                q = child.activation
                s_vec = -2.0 * q + 1.0 # Map 1 -> -1, 0 -> 1
                
                # Local Inequalities (Dlw * x <= glw)
                child.Dlw = s_vec[:, None] * W_hat
                child.glw = -(s_vec * b_hat)
                
                # Affine Map for Next Layer (Alw * x + clw)
                child.Alw = q[:, None] * W_hat
                child.clw = q * b_hat
                
                queue.append(child)

# ----------------------------------------------------------------------
# Usage Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Load specific epoch
    tree = Tree("test_experiment.h5", epoch=40)
    
    # Access properties
    print(f"Loaded Tree with {tree.L} layers.")
    print(f"Input Dimension: {tree.input_dim}")
    
    # Get regions at the final layer
    leaves = tree.get_regions_at_layer(tree.L)
    print(f"Found {len(leaves)} leaf regions.")
    
    # Check geometry of a leaf
    sample_leaf = leaves[0]
    D, g = sample_leaf.get_path_inequalities()
    print(f"Sample Leaf Constraints: {D.shape}")