import matplotlib.pyplot as plt
import seaborn as sns

def plot_layer_MI(results_dict, mi_type="I(X;W)", title_suffix="", save_path=None, H_bound=None):
    """
    Plots the evolution of Mutual Information across layers over epochs.
    
    Args:
        results_dict: Dictionary mapping a configuration label (like n_neurons) 
                      to its corresponding df_results DataFrame.
                      Example: {3: df_3, 5: df_5, 7: df_7}
        mi_type: Which MI to plot on the Y-axis ("I(X;W)" or "I(Y;W)").
        title_suffix: Optional string to append to the figure title.
        save_path: Optional filename to save the figure.
        H_bound: Optional theoretical maximum (e.g., H(X) or H(Y)) to plot as a horizontal line.
    """
    # 1. Dynamically find all unique layers across all provided dataframes
    layers = set()
    for df in results_dict.values():
        layers.update(df["layer_idx"].unique())
    layers = sorted(list(layers))
    n_layers = len(layers)
    
    # 2. Setup the figure and subplots
    fig, axes = plt.subplots(ncols=n_layers, nrows=1, figsize=(6 * n_layers, 6), sharey=True)
    
    # Ensure axes is iterable if there's only 1 layer
    if n_layers == 1:
        axes = [axes]
        
    # Set titles and basic styling for each subplot
    for ax, layer in zip(axes, layers):
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Plot theoretical maximum bound if provided
        if H_bound is not None:
            ax.axhline(y=H_bound, color='red', linestyle='--', alpha=0.5, label='Theoretical Max')
            
    axes[0].set_ylabel(f"Mutual Information {mi_type}")
    
    # 3. Create a distinct color palette for the different capacities
    colors = sns.color_palette("tab10", len(results_dict))
    
    lines = []
    labels = []
    
    # 4. Plot the data
    for (capacity, df), color in zip(results_dict.items(), colors):
        label_str = r"$d_n$ = " + str(capacity)
        
        for ax, layer in zip(axes, layers):
            # Extract data for this specific layer, ensuring chronological order
            layer_df = df[df["layer_idx"] == layer].sort_values("epoch")
            
            # Plot the line with markers
            line = ax.plot(
                layer_df["epoch"], 
                layer_df[mi_type], 
                marker="o", 
                linewidth=2,
                color=color
            )
            
            # We only need to grab the line reference once per capacity for the global legend
            if layer == layers[0]:
                lines.append(line[0])
                labels.append(label_str)
                
    # 5. Finalize layout, legend, and title
    fig.legend(lines, labels, loc='lower right', fontsize='medium', title="Capacity")
    fig.suptitle(f"Mutual Information Evolution Across Layers {title_suffix}", fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # 6. Save and show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()