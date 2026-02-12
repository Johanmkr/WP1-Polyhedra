import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
import matplotlib.colors as mcolors
import geobin_py as gb

# ==============================================================================
# 1. GEOMETRY HELPERS
# ==============================================================================

def find_interior_point(A, b):
    """
    Finds a point strictly inside the polytope Ax <= b using Chebyshev center.
    This ensures we have a valid point for HalfspaceIntersection, 
    even if the original region centroid is far outside the plotting bounds.
    """
    # We maximize radius 'r' such that: a_i*x + ||a_i||*r <= b_i
    # Variables for Linprog: [x1, x2, r]
    dim = A.shape[1]
    norm_A = np.linalg.norm(A, axis=1)
    
    # Objective: Minimize -r (Maximize r)
    c = np.zeros(dim + 1)
    c[-1] = -1 
    
    # Constraints: [A  |A|] * [x; r] <= b
    A_lp = np.hstack([A, norm_A[:, None]])
    
    # Bounds: x is free, r >= 0
    # We use a small epsilon bound for r to ensure strict interior
    bounds = [(None, None)] * dim + [(1e-6, None)]
    
    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return res.x[:-1]  # Return x part
    return None

def compute_polygon_vertices(region, bound):
    """
    Clips the unbounded region to the plotting box [-bound, bound]^2
    and returns sorted vertices.
    """
    # 1. Get Region Constraints: D*x <= g
    D, g = region.get_path_inequalities()
    dim = D.shape[1]
    
    if dim != 2:
        return None  # Only 2D supported
    
    # 2. Create Bounding Box Constraints
    # We clip the region to the view so we can see "unbounded" regions as filling the edge
    # x <= bound, x >= -bound
    I_mat = np.eye(dim)
    D_box = np.vstack([I_mat, -I_mat])
    g_box = np.full(2 * dim, bound)
    
    # 3. Combine: Region AND Box
    A_full = np.vstack([D, D_box])
    b_full = np.concatenate([g, g_box])
    
    # 4. Find a valid visual center INSIDE the box
    # The stored region.centroid might be at (1000, 1000), which is valid for the region
    # but invalid for the HalfspaceIntersection routine acting on the Box.
    interior_point = find_interior_point(A_full, b_full)
    
    if interior_point is None:
        return None # Region is empty within the viewing frame
        
    # 5. Compute Intersection
    # Scipy Halfspace format: Ax + b <= 0 -> [A, -b]
    halfspaces = np.hstack([A_full, -b_full[:, None]])
    
    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        verts = hs.intersections
        
        # 6. Sort Vertices (Counter-Clockwise)
        if len(verts) < 3: return None
        center = np.mean(verts, axis=0)
        angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
        return verts[np.argsort(angles)]
        
    except Exception:
        return None

# ==============================================================================
# 2. PLOTTING FUNCTION
# ==============================================================================

def plot_epoch_layer_grid(trees, bound=10.0, output_file="grid_visualization.png"):
    epochs = sorted(trees.keys())
    num_epochs = len(epochs)
    
    # Get depth from first tree
    first_tree = trees[epochs[0]]
    num_layers = first_tree.L
    
    # Setup Figure (Size logic: 4 inches per subplot)
    fig, axes = plt.subplots(nrows=num_layers, ncols=num_epochs, 
                             figsize=(4 * num_epochs, 4 * num_layers),
                             squeeze=True)
    
    # Setup Colors (Tab20 has 20 distinct colors)
    cmap = plt.get_cmap('tab20')
    
    print(f"Generating Plot: {num_layers} Layers x {num_epochs} Epochs...")

    for col_idx, epoch in enumerate(epochs):
        tree = trees[epoch]
        print(f"  - Processing Epoch {epoch}...")
        
        for layer_idx in range(num_layers):
            ax = axes[layer_idx, col_idx]
            layer_num = layer_idx + 1
            
            # --- 1. Formatting ---
            if layer_idx == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=14, pad=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"Layer {layer_num}", fontsize=14, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ax.set_aspect('equal')
            
            # --- 2. Get Regions & Polygons ---
            regions = tree.get_regions_at_layer(layer_num)
            
            polys = []
            facecolors = []
            
            for i, region in enumerate(regions):
                verts = compute_polygon_vertices(region, bound=bound)
                if verts is not None:
                    polys.append(verts)
                    # Cycle through colors based on index
                    facecolors.append(cmap(i % 20))
            
            # --- 3. Draw ---
            if polys:
                pc = PolyCollection(polys, 
                                    facecolors=facecolors,
                                    edgecolors='black',
                                    linewidths=0.5,
                                    alpha=0.6) # Slightly transparent
                ax.add_collection(pc)

    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()

# ==============================================================================
# Usage Example
# ==============================================================================
if __name__ == "__main__":
    # Assuming 'trees' is the dict {epoch: Tree_Object} you loaded previously
    # from the load_tree_from_h5 / Tree class code.
    
    # If not loaded, uncomment below:
    trees = {}
    for ep in [0, 10, 20, 30, 40]:
        trees[ep] = gb.Tree("test_experiment.h5", epoch=ep)

    # Check dimension before plotting
    if trees and trees[list(trees.keys())[0]].input_dim == 2:
        plot_epoch_layer_grid(trees, bound=2.2)
    else:
        print("Skipping plot: Can only plot 2D trees.")