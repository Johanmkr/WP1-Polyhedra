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
    Extracts the total and populated empirical regions at the final layer 
    and divides them by the total number of neurons in the architecture.
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
            
            # Calculate the metrics: Regions per Neuron
            df_final['tot_per_neuron'] = df_final['total_regions'] / total_neurons
            df_final['pop_per_neuron'] = df_final['populated_regions'] / total_neurons
            all_dfs.append(df_final)
                
        if not all_dfs:
            continue
            
        # Combine and aggregate statistics across all seeds
        combined_df = pd.concat(all_dfs)
        agg_df = combined_df.groupby('epoch').agg({
            'tot_per_neuron': ['mean', 'std'],
            'pop_per_neuron': ['mean', 'std']
        }).reset_index()
        
        agg_df.columns = ['epoch', 'tot_mean', 'tot_std', 'pop_mean', 'pop_std']
        plot_data[arch_tuple] = agg_df
        
    return plot_data

def plot_efficiency_subplots(base_dir_str: str):
    base_dir = Path(base_dir_str)
    
    plot_data = collect_regions_per_neuron(base_dir, target_noise=0.0)
    
    if not plot_data:
        print("No valid CSV data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    ax_tot, ax_pop = axes
    
    # Sort architectures by depth first, then width for a clean legend
    sorted_archs = sorted(plot_data.keys(), key=lambda x: (len(x), x[0]))
    
    # Styling mappings: Colors for Width, Markers for Depth
    colors = {5: 'blue', 7: 'orange', 9: 'green', 25: 'red'}
    markers = {3: 'o', 4: 's', 5: '^'}
    
    max_epoch = 0
    
    for arch in sorted_archs:
        df = plot_data[arch]
        width = arch[0]
        depth = len(arch)
        
        c = colors.get(width, 'gray')
        m = markers.get(depth, '.')
        label = f"Arch: {list(arch)}"
        
        max_epoch = max(max_epoch, int(df['epoch'].max()))
        
        # --- PLOT 1: TOTAL REGIONS PER NEURON ---
        # Removed markevery=0.1 to show markers at all data points
        ax_tot.plot(df['epoch'], df['tot_mean'], color=c, marker=m, 
                    markersize=6, linewidth=1.5, label=label, alpha=0.8)
        
        ax_tot.fill_between(df['epoch'], 
                            np.maximum(1e-5, df['tot_mean'] - df['tot_std']), 
                            df['tot_mean'] + df['tot_std'], 
                            color=c, alpha=0.05)

        # --- PLOT 2: POPULATED REGIONS PER NEURON ---
        # Removed markevery=0.1 to show markers at all data points
        ax_pop.plot(df['epoch'], df['pop_mean'], color=c, marker=m, 
                    markersize=6, linewidth=1.5, label=label, alpha=0.8)
        
        ax_pop.fill_between(df['epoch'], 
                            np.maximum(1e-5, df['pop_mean'] - df['pop_std']), 
                            df['pop_mean'] + df['pop_std'], 
                            color=c, alpha=0.05)

    # --- X-AXIS FORMATTING (Piecewise Linear Scale for early epochs) ---
    stretch = 5.0
    
    def forward(x):
        return np.where(x <= 10, x * stretch, 10 * stretch + (x - 10))

    def inverse(x):
        return np.where(x <= 10 * stretch, x / stretch, 10 + (x - 10 * stretch))

    for ax in axes:
        ax.set_xscale('function', functions=(forward, inverse))
        ax.set_xlim(left=0, right=max_epoch)
        
        # Create explicit ticks: dense for 0-10, and normal jumps thereafter
        dense_ticks = np.arange(0, 11, 2)
        standard_step = 20 if max_epoch <= 200 else 50
        standard_ticks = np.arange(standard_step, max_epoch + 1, standard_step)
        all_ticks = np.concatenate([dense_ticks, standard_ticks])
        
        ax.set_xticks(all_ticks)
        ax.set_xticklabels([str(int(t)) for t in all_ticks])
        ax.set_xlabel("Training Epochs", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Titles and Y-Labels
    ax_tot.set_ylabel("Total Regions / Total Neurons", fontsize=12)
    ax_tot.set_title("Theoretical Efficiency: Total Regions per Neuron", fontsize=14)
    
    ax_pop.set_ylabel("Populated Regions / Total Neurons", fontsize=12)
    ax_pop.set_title("Functional Efficiency: Populated Regions per Neuron", fontsize=14)

    # Place the legend outside the rightmost plot
    ax_pop.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Architectures", fontsize=10)
    
    fig.suptitle("Network Efficiency Dynamics over Training", fontsize=16, y=0.98)
    
    # Adjust layout to accommodate the external legend
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save output
    out_path = neurips_figpath / "regions_per_neuron_subplots"
    savefig(fig, str(out_path))

if __name__ == "__main__":
    from src_experiment.paths import outputs
    outputs_dir = outputs / "composite_label_noise"
    
    plot_efficiency_subplots(str(outputs_dir))