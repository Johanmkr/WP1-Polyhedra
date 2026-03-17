import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
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
    if res.success: return res.x[:-1]
    return None

def get_projection_basis(input_dim):
    if input_dim == 2: return np.eye(2)
    rng = np.random.default_rng(42)
    u = rng.standard_normal(input_dim)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(input_dim)
    v -= np.dot(u, v) * u 
    v /= np.linalg.norm(v)
    return np.vstack([u, v]).T

def compute_polygon_vertices(tree: Tree, region: Region, min_bound: float, max_bound: float, basis_matrix=None):
    D, g = tree.get_path_inequalities(region)
    if D is None or D.shape[0] == 0: return None
    if basis_matrix is not None: D = D @ basis_matrix
    dim = D.shape[1] 
    I = np.eye(dim)
    D_box = np.vstack([I, -I])
    g_box = np.concatenate([np.full(dim, max_bound), np.full(dim, -min_bound)])
    A_full = np.vstack([D, D_box])
    b_full = np.concatenate([g, g_box])
    pt = find_interior_point(A_full, b_full)
    if pt is None: return None 
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
# 2. ANIMATION FUNCTION
# ==============================================================================

def animate_experiment(h5_path, min_bound=0.0, max_bound=1.0, plot_points=False):
    path = Path(h5_path).resolve()
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"Loading data from {path.name}...")
    
    # --- A. LOAD DATA ---
    points_2d, labels = None, None
    metric_data = {}
    
    with h5py.File(path, 'r') as f:
        if 'epochs' not in f:
            print("❌ No 'epochs' found in file.")
            return
        epochs_list = sorted([int(k.split('_')[1]) for k in f['epochs'].keys()])
        
        if 'training_results' in f:
            results = f['training_results']
            for k in results.keys():
                metric_data[k] = results[k][:]
                
        if plot_points and 'points' in f and 'labels' in f:
            points_raw, labels = f['points'][:], f['labels'][:]
            if len(points_raw.shape) == 2 and points_raw.shape[0] < points_raw.shape[1]:
                points_raw = points_raw.T
            points_2d = points_raw

    trees = {}
    print("Pre-loading Trees and computing region counts...")
    for ep in epochs_list:
        try:
            t = Tree(str(path), epoch=ep)
            if t.root is not None: trees[ep] = t
        except Exception as e:
            pass

    if not trees: return

    first_tree = list(trees.values())[0]
    input_dim = first_tree.input_dim
    num_layers = min(first_tree.L, 4) # Cap at 4 layers for the 2x2 grid
    
    basis = None
    if input_dim > 2:
        basis = get_projection_basis(input_dim)
        if points_2d is not None: points_2d = points_2d @ basis

    # Precompute Region Counts for the line plot
    counts_data = {l: [] for l in range(num_layers)}
    for ep in epochs_list:
        if ep in trees:
            for l in range(num_layers):
                counts_data[l].append(len(trees[ep].get_regions_at_layer(l + 1)))

    # --- B. SETUP FIGURE LAYOUT (2x4 Grid) ---
    # Made the overall canvas slightly more square (14x10 instead of 16x9)
    fig = plt.figure(figsize=(14, 10))
    
    # width_ratios changed: the right two columns (1.6) now take up significantly 
    # more space than the left two columns (1.0).
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1.6, 1.6], wspace=0.25, hspace=0.2)
    
    # Left Side: Metrics (Row 0, Cols 0-1) and Region Counts (Row 1, Cols 0-1)
    ax_metrics = fig.add_subplot(gs[0, 0:2])
    ax_acc = ax_metrics.twinx()
    ax_counts = fig.add_subplot(gs[1, 0:2])
    
    # Right Side: Layers (2x2 Grid in Cols 2-3)
    ax_layers = [
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])
    ]

    metric_epochs = np.arange(1, len(epochs_list) + 1)

    # --- C. INITIALIZE METRIC & COUNT PLOTS ---
    def get_style(metric_name):
        return '--' if 'test' in metric_name or 'val' in metric_name else '-'

    # Metrics Axis
    lines_metrics = {}
    ax_metrics.set_xlim(1, len(epochs_list))
    all_loss = [metric_data[k] for k in metric_data if 'loss' in k]
    if all_loss: ax_metrics.set_ylim(max(0, np.min(all_loss) - 0.1), np.max(all_loss) + 0.1)
    ax_metrics.set_ylabel("Loss", color='blue')
    ax_metrics.tick_params(axis='y', labelcolor='blue')
    ax_metrics.set_title("Training Metrics", fontweight='bold')
    
    all_acc = [metric_data[k] for k in metric_data if 'acc' in k]
    if all_acc: ax_acc.set_ylim(0, 1.05)
    ax_acc.set_ylabel("Accuracy", color='red')
    ax_acc.tick_params(axis='y', labelcolor='red')

    for k in metric_data.keys():
        color = 'blue' if 'loss' in k else 'red' if 'acc' in k else 'gray'
        ax_target = ax_metrics if 'loss' in k or color == 'gray' else ax_acc
        line, = ax_target.plot([], [], color=color, linestyle=get_style(k), label=k)
        lines_metrics[k] = line
        
    lines1, labels1 = ax_metrics.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax_metrics.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Counts Axis
    lines_counts = []
    ax_counts.set_xlim(1, len(epochs_list))
    max_count = max([max(counts_data[l]) for l in range(num_layers)]) if counts_data[0] else 10
    ax_counts.set_ylim(0, max_count * 1.1)
    ax_counts.set_xlabel("Epochs")
    ax_counts.set_ylabel("Number of Regions")
    ax_counts.set_title("Linear Regions per Layer", fontweight='bold')
    ax_counts.grid(True, alpha=0.3)
    
    layer_colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, num_layers))
    for l in range(num_layers):
        line, = ax_counts.plot([], [], color=layer_colors[l], linewidth=2, label=f"Layer {l+1}")
        lines_counts.append(line)
    ax_counts.legend(loc="center right")

    # --- D. INITIALIZE LAYER PLOTS ---
    spatial_cmap = plt.get_cmap('hsv') # Cyclic colormap for consistent spatial coloring
    domain_center_x = (min_bound + max_bound) / 2
    domain_center_y = (min_bound + max_bound) / 2

    for i, ax in enumerate(ax_layers):
        if i < num_layers:
            ax.set_xlim(min_bound, max_bound)
            ax.set_ylim(min_bound, max_bound)
            ax.set_aspect('equal')
            ax.set_title(f"Layer {i + 1}", fontsize=11)
            # Make tighter by removing tick labels on the spatial plots
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    point_colors = ['#FF0000', '#00FFFF', '#39FF14', '#FF00FF', '#FFFF00', '#FFFFFF']
    point_markers = ['o', 's', '^', 'D', 'P', '*'] 

    # --- E. ANIMATION UPDATE FUNCTION ---
    def update(frame_idx):
        ep = epochs_list[frame_idx]
        fig.suptitle(f"Epoch {ep}", fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Update Line Plots
        current_x = metric_epochs[:frame_idx + 1]
        for k, line in lines_metrics.items():
            line.set_data(current_x, metric_data[k][:frame_idx + 1])
            
        for l, line in enumerate(lines_counts):
            line.set_data(current_x, counts_data[l][:frame_idx + 1])

        # 2. Update Layers
        tree = trees.get(ep)
        if tree is None: return

        for layer_idx in range(num_layers):
            ax = ax_layers[layer_idx]
            [c.remove() for c in ax.collections]
            [l.remove() for l in ax.lines]
            
            regions = tree.get_regions_at_layer(layer_idx + 1)
            polys = []
            facecolors = []
            
            for region in regions:
                verts = compute_polygon_vertices(tree, region, min_bound=min_bound, max_bound=max_bound, basis_matrix=basis)
                if verts is not None:
                    polys.append(verts)
                    
                    # Spatial Coloring Logic
                    center = np.mean(verts, axis=0)
                    angle = np.arctan2(center[1] - domain_center_y, center[0] - domain_center_x)
                    norm_angle = (angle + np.pi) / (2 * np.pi) # Normalize -pi..pi to 0..1
                    facecolors.append(spatial_cmap(norm_angle))
            
            if polys:
                pc = PolyCollection(polys, facecolors=facecolors, edgecolors='black', 
                                    linewidths=0.5, alpha=0.6, zorder=1)
                ax.add_collection(pc)

            if points_2d is not None and labels is not None:
                for idx, lbl in enumerate(np.unique(labels)):
                    mask = (labels == lbl)
                    ax.plot(points_2d[mask, 0], points_2d[mask, 1], 
                            color=point_colors[idx % len(point_colors)], 
                            marker=point_markers[idx % len(point_markers)], 
                            linestyle='None', markersize=1.5, alpha=0.6, zorder=4)

    # --- F. RENDER AND SAVE ---
    print(f"Generating animation across {len(epochs_list)} epochs...")
    # Adjusted margins to fit the new aspect ratio perfectly
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.98)
    
    anim = FuncAnimation(fig, update, frames=len(epochs_list), interval=200, blit=False)
    out_file = path.parent / f"{path.stem}_animation.mp4"
    
    try:
        anim.save(out_file, writer='ffmpeg', fps=20, dpi=200)
        print(f"✅ Animation successfully saved to: {out_file}")
    except Exception as e:
        print(f"❌ Error saving animation. Do you have FFmpeg installed? Error details:\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate training metrics and geometric regions.")
    parser.add_argument("file", type=str, help="HDF5 file path")
    parser.add_argument("--min_bound", type=float, default=-1.0, help="Minimum plot view bound (default: -1.0)")
    parser.add_argument("--max_bound", type=float, default=1.0, help="Maximum plot view bound (default: 1.0)")
    parser.add_argument("--points", action="store_true", help="Plot scatter points from the HDF5 file")
    
    args = parser.parse_args()
    animate_experiment(args.file, args.min_bound, args.max_bound, args.points)