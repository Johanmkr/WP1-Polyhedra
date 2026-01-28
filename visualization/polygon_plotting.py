

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from intvalpy import lineqs


def find_vertices_from_regions(regions, bound=10):
    vertices_list = []
    for region in regions:
        D, g = region.get_path_inequalities()
        
        try:
            # Use a small tolerance or check feasibility first
            verts = lineqs(-D, -g, bounds=[[-bound,-bound], [bound,bound]], show=False, size=(5,5))
            vertices_list.append(verts)
        except (IndexError, ValueError) as e:
            # print(D, g)
            print("Warning: Polytope has no vertices or is empty. Skipping visualization.")
            continue
    return vertices_list

def plot_epoch_layer_grid(trees,  bound=50):
    epochs = sorted(trees.keys())
    num_epochs = len(epochs)
    num_layers = trees[0].L
    # 1. Setup the figure: Rows = Layers, Cols = Epochs
    fig, axes = plt.subplots(num_layers, num_epochs, 
                             figsize=(num_epochs * 3, num_layers * 3), 
                             constrained_layout=True)
    
    # Standardize axes to a 2D array for consistent indexing [row, col]
    if num_layers == 1 and num_epochs == 1:
        axes = np.array([[axes]])
    elif num_layers == 1:
        axes = axes[np.newaxis, :]
    elif num_epochs == 1:
        axes = axes[:, np.newaxis]

    # 2. Iterate through Epochs (Columns)
    for col, epoch in enumerate(epochs):
        tree = trees[epoch]
        
        # 3. Iterate through Layers (Rows)
        for row in range(num_layers):
            layer_idx = row + 1
            ax = axes[row, col]
            
            # Access existing regions at this specific layer for this specific root
            regions = tree.get_regions_at_layer(layer=layer_idx)
            
            vertices_list = find_vertices_from_regions(regions, bound=20000)
            
            # 4. Draw Polygons
            for vert in vertices_list:
                # Basic boundary check for the 2D slice
                within_boundaries = any(-bound <= coord[0] <= bound and 
                                        -bound <= coord[1] <= bound for coord in vert)
                
                if within_boundaries:
                    # Plot the polygon
                    poly = Polygon(xy=vert, 
                                   facecolor=np.random.rand(3,), 
                                   edgecolor="black", 
                                   alpha=0.6, 
                                   linewidth=0.5)
                    ax.add_patch(poly)

            # Formatting each subplot
            ax.set_xlim([-bound, bound])
            ax.set_ylim([-bound, bound])
            ax.set_aspect('equal')
            
            # Labels: Epochs on top, Layers on the left
            if row == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=14, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=14, fontweight='bold')
            
            # Remove tick clutter for inner plots
            if row < num_layers - 1:
                ax.set_xticks([])
            if col > 0:
                ax.set_yticks([])

    return fig, axes