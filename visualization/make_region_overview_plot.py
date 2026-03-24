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
    from src_experiment.paths import neurips_figpath
    from src_experiment.utils import savefig
except ImportError:
    print("❌ Error: Could not import 'geobin_py.reconstruction'.")
    print("   Ensure the file 'geobin_py/reconstruction.py' exists relative to the project root.")
    sys.exit(1)

# ==============================================================================
# 1. GEOMETRY HELPERS 
# ==============================================================================

def find_interior_point(A, b):
    dim = A.shape[1]
    norm_A = np.linalg.norm(A, axis=1)
    
    c = np.zeros(dim + 1)
    c[-1] = -1 
    
    A_lp = np.hstack([A, norm_A[:, None]])
    bounds = [(None, None)] * dim + [(1e-6, None)]
    
    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return res.x[:-1] 
    return None

def get_projection_basis(input_dim):
    if input_dim == 2:
        return np.eye(2)
    
    rng = np.random.default_rng(42) 
    u = rng.standard_normal(input_dim)
    u /= np.linalg.norm(u)
    
    v = rng.standard_normal(input_dim)
    v -= np.dot(u, v) * u 
    v /= np.linalg.norm(v)
    
    return np.vstack([u, v]).T

def compute_polygon_vertices(tree: Tree, region: Region, min_bound: float, max_bound: float, basis_matrix=None):
    D, g = tree.get_path_inequalities(region)
    if D is None or D.shape[0] == 0:
        return None

    if basis_matrix is not None:
        D = D @ basis_matrix
    
    dim = D.shape[1] 
    
    I = np.eye(dim)
    D_box = np.vstack([I, -I])
    g_box = np.concatenate([np.full(dim, max_bound), np.full(dim, -min_bound)])
    
    A_full = np.vstack([D, D_box])
    b_full = np.concatenate([g, g_box])
    
    pt = find_interior_point(A_full, b_full)
    if pt is None:
        return None 
        
    halfspaces = np.hstack([A_full, -b_full[:, None]])
    
    try:
        hs = HalfspaceIntersection(halfspaces, pt)
        verts = hs.intersections
        
        if len(verts) < 3: return None
        center = np.mean(verts, axis=0)
        angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
        return verts[np.argsort(angles)]
    except Exception:
        return None

# ==============================================================================
# 2. PLOTTING FUNCTIONS
# ==============================================================================

def plot_last_epoch_layers(h5_path, min_bound=-1.0, max_bound=1.0, plot_points=False, sp="sometitle"):
    """Plots the filled regions and hierarchical boundaries for the final epoch."""
    path = Path(h5_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"Scanning {path.name} for final epoch layers...")
    points, labels = None, None
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            if 'epochs' not in f: raise KeyError
            all_epochs = sorted([int(k.split('_')[1]) for k in f['epochs'].keys()])
            last_epoch = all_epochs[-1]
            
            if plot_points and 'points' in f and 'labels' in f:
                points_raw = f['points'][:]
                labels = f['labels'][:]
                if len(points_raw.shape) == 2 and points_raw.shape[0] < points_raw.shape[1]:
                    points_raw = points_raw.T
                points = points_raw
    except Exception as e:
        print(f"❌ Could not read epochs. Error: {e}")
        return

    try:
        tree = Tree(str(path), epoch=last_epoch)
        if tree.root is None: return
    except Exception as e:
        print(f"❌ Failed to load epoch {last_epoch}: {e}")
        return

    input_dim = tree.input_dim
    layers_to_plot = max(1, tree.L - 1) # Exclude the last layer
    
    basis = get_projection_basis(input_dim) if input_dim > 2 else None
    slice_msg = f"Random 2D Slice of {input_dim}D Space" if input_dim > 2 else "2D Input Space"
    points_2d = points @ basis if (points is not None and basis is not None) else points
    
    fig, axes = plt.subplots(nrows=2, ncols=layers_to_plot, figsize=(4 * layers_to_plot, 8), squeeze=False) 
    
    cmap = plt.get_cmap('tab20')
    layer_boundary_colors = ["#f80000", "#00ff00","#001aff", "#ec00d9", "#ff7700" ]

    # Pre-compute all polygons
    all_layer_polys = []
    all_layer_facecolors = []
    for layer_idx in range(layers_to_plot):
        regions = tree.get_regions_at_layer(layer_idx + 1)
        polys, f_colors = [], []
        for i, region in enumerate(regions):
            verts = compute_polygon_vertices(tree, region, min_bound, max_bound, basis)
            if verts is not None:
                polys.append(verts)
                f_colors.append(cmap(i % 20))
        all_layer_polys.append(polys)
        all_layer_facecolors.append(f_colors)

    for layer_idx in range(layers_to_plot):
        ax_filled = axes[0, layer_idx] 
        ax_bounds = axes[1, layer_idx] 
        
        ax_filled.tick_params(axis='both', left=(layer_idx == 0), bottom=False, labelleft=(layer_idx == 0), labelbottom=False)
        ax_bounds.tick_params(axis='both', left=(layer_idx == 0), bottom=True, labelleft=(layer_idx == 0), labelbottom=True)

        ax_filled.set_title(f"Layer {layer_idx + 1}", fontsize=12, fontweight='bold')
        if layer_idx == 0:
            ax_filled.set_ylabel("Filled Regions", fontsize=12, fontweight='bold')
            ax_bounds.set_ylabel("Boundaries", fontsize=12, fontweight='bold')
        
        for ax in (ax_filled, ax_bounds):
            ax.set_xlim(min_bound, max_bound)
            ax.set_ylim(min_bound, max_bound)
            ax.set_aspect('equal')
        
        polys = all_layer_polys[layer_idx]
        facecolors = all_layer_facecolors[layer_idx]
        
        if polys:
            # Row 1: Colored Face, Black Edges
            pc_filled = PolyCollection(polys, facecolors=facecolors, edgecolors='black', 
                                       linewidths=0.5, alpha=0.6, zorder=2)
            ax_filled.add_collection(pc_filled)
            
            # Row 2: Boundary Lineage (Restored Original Logic)
            c_color = layer_boundary_colors[layer_idx % len(layer_boundary_colors)]
            
            # 1. Base Layer: Draw current polygons with white faces and THICK current-layer edges
            pc_bounds = PolyCollection(polys, facecolors='white', edgecolors=c_color, 
                                       linewidths=1.0, alpha=1.0, zorder=2)
            ax_bounds.add_collection(pc_bounds)
            
            # 2. Overlay older boundaries THIN and ON TOP (highest zorder for oldest layers)
            for prev_idx in range(layer_idx - 1, -1, -1):
                prev_polys = all_layer_polys[prev_idx]
                p_color = layer_boundary_colors[prev_idx % len(layer_boundary_colors)]
                
                # Higher z-order guarantees the oldest lines strictly cut through the newer ones
                stacking_zorder = 3 + (layer_idx - prev_idx)
                
                pc_prev = PolyCollection(prev_polys, facecolors='none', edgecolors=p_color, 
                                         linewidths=0.4, alpha=1.0, zorder=stacking_zorder)
                ax_bounds.add_collection(pc_prev)
        else:
            for ax in (ax_filled, ax_bounds):
                ax.text(0, 0, "Empty/No Slice", ha='center', va='center', fontsize=10)

        if points_2d is not None and labels is not None:
            point_colors = ['#FF0000', '#00FFFF', '#39FF14', '#FF00FF', '#FFFF00', '#000000']
            point_markers = ['o', 's', '^', 'D', 'P', '*'] 
            for idx, lbl in enumerate(np.unique(labels)):
                mask = (labels == lbl)
                c, m = point_colors[idx % 6], point_markers[idx % 6]
                for ax in (ax_filled, ax_bounds):
                    ax.scatter(points_2d[mask, 0], points_2d[mask, 1], c=c, marker=m, s=2, alpha=0.8, zorder=10)

    # fig.suptitle(f"Final Epoch ({last_epoch}) Regions per Layer\n{slice_msg}", fontsize=14, y=0.99)
    plt.tight_layout()
    # plt.savefig(neurips_figpath / sp, dpi=300)
    savefig(fig, neurips_figpath / sp)
    
    print("✅ Saved Last Epoch Plot.")


def plot_epoch_grid(h5_path, min_bound=-1.0, max_bound=1.0, plot_points=False, sp="sometitle"):
    """Plots a grid showing the evolution of all layers (excluding the last) across multiple epochs."""
    path = Path(h5_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"Scanning {path.name} for epoch grid...")
    points, labels = None, None
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            if 'epochs' not in f: raise KeyError
            all_epochs = sorted([int(k.split('_')[1]) for k in f['epochs'].keys()])
            
            if plot_points and 'points' in f and 'labels' in f:
                points_raw, labels = f['points'][:], f['labels'][:]
                points = points_raw.T if (len(points_raw.shape) == 2 and points_raw.shape[0] < points_raw.shape[1]) else points_raw
    except Exception as e:
        print(f"❌ Could not read epochs. Error: {e}")
        return

    # Sample up to 6 epochs to fit nicely in a grid
    if len(all_epochs) > 6:
        indices = np.linspace(0, len(all_epochs)-1, 6, dtype=int)
        epochs_to_plot = sorted(list(set([all_epochs[i] for i in indices])))
    else:
        epochs_to_plot = all_epochs

    trees = {}
    for ep in epochs_to_plot:
        try:
            t = Tree(str(path), epoch=ep)
            if t.root is not None: trees[ep] = t
        except Exception: pass

    if not trees: return

    first_tree = list(trees.values())[0]
    input_dim = first_tree.input_dim
    # Exclude the last layer dynamically here as requested
    layers_to_plot = max(1, first_tree.L - 1) 
    
    basis = get_projection_basis(input_dim) if input_dim > 2 else None
    slice_msg = f"Random 2D Slice of {input_dim}D Space" if input_dim > 2 else "2D Input Space"
    points_2d = points @ basis if (points is not None and basis is not None) else points
    
    num_epochs = len(epochs_to_plot)
    fig, axes = plt.subplots(nrows=layers_to_plot, ncols=num_epochs, 
                             figsize=(3.5 * num_epochs, 3.5 * layers_to_plot), squeeze=False)
    cmap = plt.get_cmap('tab20')

    for col_idx, ep in enumerate(epochs_to_plot):
        if ep not in trees: continue
        tree = trees[ep]
        print(f"  > Rendering Epoch {ep}...")
        
        for layer_idx in range(layers_to_plot):
            ax = axes[layer_idx, col_idx]
            
            # Axis hiding logic
            ax.tick_params(axis='both', left=(col_idx == 0), bottom=(layer_idx == layers_to_plot - 1),
                           labelleft=(col_idx == 0), labelbottom=(layer_idx == layers_to_plot - 1))

            if layer_idx == 0: ax.set_title(f"Epoch {ep}", fontsize=12, fontweight='bold')
            if col_idx == 0: ax.set_ylabel(f"Layer {layer_idx + 1}", fontsize=12, fontweight='bold')
            
            ax.set_xlim(min_bound, max_bound)
            ax.set_ylim(min_bound, max_bound)
            ax.set_aspect('equal')
            
            regions = tree.get_regions_at_layer(layer_idx + 1)
            polys, facecolors = [], []
            for i, region in enumerate(regions):
                verts = compute_polygon_vertices(tree, region, min_bound, max_bound, basis)
                if verts is not None:
                    polys.append(verts)
                    facecolors.append(cmap(i % 20))
            
            if polys:
                pc = PolyCollection(polys, facecolors=facecolors, edgecolors='black', linewidths=0.5, alpha=0.6)
                ax.add_collection(pc)
            else:
                ax.text(0, 0, "Empty/No Slice", ha='center', va='center', fontsize=8)

            if points_2d is not None and labels is not None:
                point_colors = ['#FF0000', '#00FFFF', '#39FF14', '#FF00FF', '#FFFF00', '#FFFFFF']
                point_markers = ['o', 's', '^', 'D', 'P', '*'] 
                for idx, lbl in enumerate(np.unique(labels)):
                    mask = (labels == lbl)
                    c, m = point_colors[idx % 6], point_markers[idx % 6]
                    ax.scatter(points_2d[mask, 0], points_2d[mask, 1], c=c, marker=m, s=1, alpha=0.6, zorder=4)

    # fig.suptitle(f"Geometric Regions Evolution\n{slice_msg}", fontsize=16, y=1.02)
    plt.tight_layout()
    # plt.savefig(neurips_figpath / sp, dpi=300)
    savefig(fig, neurips_figpath / sp)
    print(f"✅ Evolution Grid")


if __name__ == "__main__":
    # If run via command line, try to parse arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("file", type=str, help="HDF5 file path")
        parser.add_argument("--min_bound", type=float, default=-1.3, help="Minimum plot view bound (default: -1.3)")
        parser.add_argument("--max_bound", type=float, default=1.3, help="Maximum plot view bound (default: 1.3)")
        parser.add_argument("--points", action="store_true", help="Plot scatter points from the HDF5 file")
        parser.add_argument("--grid", action="store_true", help="Generate the epoch evolution grid instead of the last epoch plot")
        args = parser.parse_args()
        
        if args.grid:
            plot_epoch_grid(args.file, args.min_bound, args.max_bound, args.points)
        else:
            plot_last_epoch_layers(args.file, args.min_bound, args.max_bound, args.points, sp=Path(args.file).stem + "_final_regions.pdf")
            plt.show()
            
    else:
        # Fallback to the hardcoded examples for your test files
        large_moons_file = "outputs/moons_test2/moons_test2.h5"
        small_circles_file = "outputs/circles_test/circles_test.h5"
        

        plot_last_epoch_layers(small_circles_file, -1.75, 1.75, sp="small_circles_regions.pdf")

        plot_epoch_grid(small_circles_file, -1.75, 1.75, plot_points=True, sp="small_circles_regions_full.pdf")
        
        plot_last_epoch_layers(large_moons_file, -1.75, 1.75, sp="large_moons_regions.pdf")

        plot_epoch_grid(large_moons_file, -1.75, 1.75, plot_points=True, sp="large_moons_regions_full.pdf")
        
        plt.show()