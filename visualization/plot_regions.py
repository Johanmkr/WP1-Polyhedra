import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from pathlib import Path

# Add project root to sys.path to ensure we can import geobin_py
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from geobin_py.reconstruction import Tree, Region
except ImportError:
    print("❌ Error: Could not import 'geobin_py.reconstruction'.")
    print("   Ensure the file 'geobin_py/reconstruction.py' exists relative to the project root.")
    sys.exit(1)

# ==============================================================================
# 1. GEOMETRY HELPERS (Halfspace Intersection)
# ==============================================================================

def find_interior_point(A, b):
    """
    Finds a point strictly inside the polytope Ax <= b using Chebyshev center via Linear Programming.
    Maximize r subject to: a_i*x + ||a_i||*r <= b_i
    """
    dim = A.shape[1]
    norm_A = np.linalg.norm(A, axis=1)
    
    # Objective: Minimize -r (Maximize r)
    # Variables: [x1, x2, ..., x_dim, r]
    c = np.zeros(dim + 1)
    c[-1] = -1 
    
    # Constraints: [A  |A|] * [x; r] <= b
    A_lp = np.hstack([A, norm_A[:, None]])
    
    # Bounds: x is free, r >= epsilon (strictly interior)
    bounds = [(None, None)] * dim + [(1e-6, None)]
    
    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return res.x[:-1]  # Return x part
    return None

def get_projection_basis(input_dim):
    """
    Returns a (Dim, 2) matrix U defining a random 2D slice: x = U * y.
    If dim=2, returns Identity.
    """
    if input_dim == 2:
        return np.eye(2)
    
    rng = np.random.default_rng(42) # Fixed seed for consistent slicing across epochs
    u = rng.standard_normal(input_dim)
    u /= np.linalg.norm(u)
    
    v = rng.standard_normal(input_dim)
    v -= np.dot(u, v) * u # Orthogonalize
    v /= np.linalg.norm(v)
    
    return np.vstack([u, v]).T

def compute_polygon_vertices(region: Region, bound: float, basis_matrix=None):
    """
    Computes vertices of the region intersected with the viewing box [-bound, bound]^2.
    Handles High-D inputs by projecting constraints onto the basis_matrix.
    """
    # 1. Get Inequalities: D * x <= g
    D, g = region.get_path_inequalities()
    if D is None or D.shape[0] == 0:
        return None

    # 2. Project High-D constraints to 2D slice if needed
    if basis_matrix is not None:
        D = D @ basis_matrix
    
    dim = D.shape[1] # Should be 2 now
    
    # 3. Create Bounding Box Constraints [-bound, bound]
    I = np.eye(dim)
    D_box = np.vstack([I, -I])
    g_box = np.full(2 * dim, bound)
    
    # 4. Combine: Region AND Box
    A_full = np.vstack([D, D_box])
    b_full = np.concatenate([g, g_box])
    
    # 5. Find Interior Point (Chebyshev Center)
    pt = find_interior_point(A_full, b_full)
    
    if pt is None:
        return None # Region doesn't intersect our viewing slice/box
        
    # 6. Compute Intersection using Scipy
    halfspaces = np.hstack([A_full, -b_full[:, None]])
    
    try:
        hs = HalfspaceIntersection(halfspaces, pt)
        verts = hs.intersections
        
        # 7. Sort Vertices Counter-Clockwise
        if len(verts) < 3: return None
        center = np.mean(verts, axis=0)
        angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
        return verts[np.argsort(angles)]
        
    except Exception:
        # Fails if region is degenerate or numerical issues
        return None

# ==============================================================================
# 2. PLOTTING FUNCTION
# ==============================================================================

def plot_grid(h5_path, bound=4.0, plot_points=False):
    path = Path(h5_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    # 1. Scan Epochs & Load Points
    print(f"Scanning {path.name}...")
    points = None
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            if 'epochs' not in f: raise KeyError
            all_epochs = sorted([int(k.split('_')[1]) for k in f['epochs'].keys()])
            
            # Extract points if requested
            if plot_points:
                if 'points' in f and 'labels' in f:
                    points_raw = f['points'][:]
                    labels = f['labels'][:] # <--- Load labels
                    
                    if len(points_raw.shape) == 2 and points_raw.shape[0] < points_raw.shape[1]:
                        points_raw = points_raw.T
                        
                    points = points_raw
                else:
                    print("⚠️ '--points' flag was used, but 'points' or 'labels' were missing.")
                    
    except Exception as e:
        print(f"❌ Could not read epochs or points. Has the Julia script run? Error: {e}")
        return

    if len(all_epochs) > 6:
        indices = np.linspace(0, len(all_epochs)-1, 6, dtype=int)
        epochs_to_plot = sorted(list(set([all_epochs[i] for i in indices])))
    else:
        epochs_to_plot = all_epochs

    # 2. Load Trees
    trees = {}
    for ep in epochs_to_plot:
        try:
            t = Tree(str(path), epoch=ep)
            if t.root is not None:
                trees[ep] = t
        except Exception as e:
            print(f"  Skipping Epoch {ep}: {e}")

    if not trees:
        print("❌ No valid trees loaded.")
        return

    first_tree = list(trees.values())[0]
    input_dim = first_tree.input_dim
    num_layers = first_tree.L
    
    basis = None
    slice_msg = "2D Input Space"
    if input_dim > 2:
        basis = get_projection_basis(input_dim)
        slice_msg = f"Random 2D Slice of {input_dim}D Space"
        
    # 3. Project points to 2D slice if necessary
    points_2d = None
    if points is not None:
        if input_dim > 2 and basis is not None:
            points_2d = points @ basis
        else:
            points_2d = points
    
    # 4. Initialize Plot
    num_epochs = len(epochs_to_plot)
    fig, axes = plt.subplots(nrows=num_layers, ncols=num_epochs, 
                             figsize=(3.5 * num_epochs, 3.5 * num_layers),
                             squeeze=False)
    
    cmap = plt.get_cmap('tab20')

    # 5. Render Loop
    for col_idx, ep in enumerate(epochs_to_plot):
        if ep not in trees: continue
        tree = trees[ep]
        
        print(f"  > Rendering Epoch {ep}...")
        
        for layer_idx in range(num_layers):
            ax = axes[layer_idx, col_idx]
            
            # --- TICK LOGIC ---
            if col_idx != 0:
                ax.set_yticklabels([])
            if layer_idx != num_layers - 1:
                ax.set_xticklabels([])
                
            ax.tick_params(axis='both', which='both', 
                           left=(col_idx == 0), 
                           bottom=(layer_idx == num_layers - 1),
                           labelleft=(col_idx == 0), 
                           labelbottom=(layer_idx == num_layers - 1))

            # Formatting
            if layer_idx == 0:
                ax.set_title(f"Epoch {ep}", fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx + 1}", fontsize=12, fontweight='bold')
            
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ax.set_aspect('equal')
            
            # Fetch Regions
            regions = tree.get_regions_at_layer(layer_idx + 1)
            
            polys = []
            facecolors = []
            
            for i, region in enumerate(regions):
                verts = compute_polygon_vertices(region, bound=bound, basis_matrix=basis)
                if verts is not None:
                    polys.append(verts)
                    facecolors.append(cmap(i % 20))
            
            if polys:
                pc = PolyCollection(polys, 
                                    facecolors=facecolors,
                                    edgecolors='black',
                                    linewidths=0.5,
                                    alpha=0.6,
                                    zorder=1) # Render regions at bottom
                ax.add_collection(pc)
            else:
                ax.text(0, 0, "Empty/No Slice", ha='center', va='center', fontsize=8)

            # Scatter the points if requested
            if points_2d is not None:
                # 1. Define distinct, high-contrast colors and shapes
                point_colors = ['#FF0000', '#00FFFF', '#39FF14', '#FF00FF', '#FFFF00', '#FFFFFF']
                point_markers = ['o', 's', '^', 'D', 'P', '*'] 
                
                unique_labels = np.unique(labels)
                
                # 2. Plot each class separately
                for idx, lbl in enumerate(unique_labels):
                    mask = (labels == lbl)
                    
                    color = point_colors[idx % len(point_colors)]
                    marker = point_markers[idx % len(point_markers)]
                    
                    ax.scatter(points_2d[mask, 0], points_2d[mask, 1], 
                               c=color, 
                               marker=marker,
                               s=1,                # <--- CHANGED: Reduced from 25 to 8
                               alpha=0.6,          
                               zorder=4,           
                            #    edgecolors='black', 
                               linewidths=0.5     # <--- CHANGED: Thinner border (was 1.0) so it doesn't swallow the color
                    )

    # 6. Save
    fig.suptitle(f"Geometric Regions Evolution\n{slice_msg}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    out_file = path.parent / f"{path.stem}_regions_grid.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="HDF5 file path")
    parser.add_argument("--bound", type=float, default=4.0, help="Plot view bound (default: 4.0)")
    parser.add_argument("--points", action="store_true", help="Plot scatter points from the HDF5 file")
    args = parser.parse_args()
    
    plot_grid(args.file, args.bound, args.points)