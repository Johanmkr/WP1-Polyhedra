import os
import sys
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the project root is in the path to import src_experiment
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src_experiment.utils import savefig
from src_experiment.paths import neurips_figpath

def parse_folder_name(folder_name: str):
    """Parses 'n0.0_[7, 7, 7]' into noise=0.0, arch=[7, 7, 7]"""
    try:
        parts = folder_name.split('_', 1)
        noise_str = parts[0].replace('n', '')
        noise = float(noise_str)
        arch = ast.literal_eval(parts[1])
        return noise, arch
    except Exception as e:
        return None, None

def collect_regions_per_neuron(base_dir: Path, target_noise: float = 0.0):
    """
    Extracts the total empirical regions at the final layer and divides it 
    by the total number of neurons in the architecture.
    """
    plot_data = {}
    seeds = [101, 102, 103, 104, 105]
    
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
            
        noise, arch = parse_folder_name(folder.name)
        
        if noise != target_noise or not arch:
            continue
            
        arch_tuple = tuple(arch)
        total_neurons = sum(arch)
        all_dfs = []
        
        for seed in seeds:
            results_csv = folder / f"evaluated_metrics_seed_{seed}.csv"
            
            if not results_csv.exists():
                continue
                
            df = pd.read_csv(results_csv)
            if df.empty:
                continue
                
            # Filter for the final hidden layer to get the network's total regions
            final_layer_idx = df['layer_idx'].max()
            df_final = df[df['layer_idx'] == final_layer_idx].copy()
            
            # Calculate the metric: Regions per Neuron
            df_final['regions_per_neuron'] = df_final['total_regions'] / total_neurons
            all_dfs.append(df_final)
                
        if not all_dfs:
            continue
            
        # Combine and aggregate statistics across all seeds
        combined_df = pd.concat(all_dfs)
        agg_df = combined_df.groupby('epoch').agg({
            'regions_per_neuron': ['mean', 'std']
        }).reset_index()
        
        agg_df.columns = ['epoch', 'mean', 'std']
        plot_data[arch_tuple] = agg_df
        
    return plot_data

def plot_efficiency(base_dir_str: str):
    base_dir = Path(base_dir_str)
    
    plot_data = collect_regions_per_neuron(base_dir, target_noise=0.0)
    
    if not plot_data:
        print("No valid CSV data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Sort architectures by depth first, then width for a clean legend
    sorted_archs = sorted(plot_data.keys(), key=lambda x: (len(x), x[0]))
    
    # Styling mappings
    colors = {5: 'blue', 7: 'orange', 9: 'green', 25: 'red'}
    styles = {3: '-', 4: '--', 5: ':'}
    
    max_epoch = 0
    
    for arch in sorted_archs:
        df = plot_data[arch]
        width = arch[0]
        depth = len(arch)
        
        c = colors.get(width, 'gray')
        s = styles.get(depth, '-.')
        
        label = f"Arch: {list(arch)}"
        
        # Track max epoch to dynamically set axes limits later
        max_epoch = max(max_epoch, int(df['epoch'].max()))
        
        # Plot the mean line
        ax.plot(df['epoch'], df['mean'], color=c, linestyle=s, linewidth=2, label=label)
        
        # Add shaded std deviation
        ax.fill_between(df['epoch'], 
                        np.maximum(1e-5, df['mean'] - df['std']), 
                        df['mean'] + df['std'], 
                        color=c, alpha=0.05)

    # --- X-AXIS FORMATTING (Piecewise Linear Scale for early epochs) ---
    # We apply a stretch factor to epochs 0-10 so they take up more visual space.
    # A stretch of 5.0 means epochs 0-10 occupy the same width as 50 normal epochs.
    stretch = 5.0
    
    def forward(x):
        return np.where(x <= 10, x * stretch, 10 * stretch + (x - 10))

    def inverse(x):
        return np.where(x <= 10 * stretch, x / stretch, 10 + (x - 10 * stretch))

    ax.set_xscale('function', functions=(forward, inverse))
    ax.set_xlim(left=0, right=max_epoch)
    
    # Create explicit ticks: dense for 0-10, and normal jumps thereafter
    dense_ticks = np.arange(0, 11, 2)  # 0, 2, 4, 6, 8, 10
    standard_step = 20 if max_epoch <= 200 else 50
    standard_ticks = np.arange(standard_step, max_epoch + 1, standard_step)
    
    all_ticks = np.concatenate([dense_ticks, standard_ticks])
    
    ax.set_xticks(all_ticks)
    ax.set_xticklabels([str(int(t)) for t in all_ticks])

    # ax.set_yscale('log') # Uncomment if you also want logarithmic Y axis
    ax.set_xlabel("Training Epochs", fontsize=12)
    ax.set_ylabel("Total Regions / Total Neurons", fontsize=12)
    ax.set_title("Network Efficiency: Regions Created per Neuron Over Time", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Place the legend outside the plot to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Architectures", fontsize=10)
    
    # Adjust layout to accommodate the external legend
    fig.tight_layout()
    
    # Save output
    out_path = neurips_figpath / "regions_per_neuron_dynamics"
    savefig(fig, str(out_path))

if __name__ == "__main__":
    from src_experiment.paths import outputs
    outputs_dir = outputs / "composite_label_noise"
    
    plot_efficiency(str(outputs_dir))