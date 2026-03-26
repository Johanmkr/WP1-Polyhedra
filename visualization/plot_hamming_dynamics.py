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
    try:
        parts = folder_name.split('_', 1)
        noise_str = parts[0].replace('n', '')
        noise = float(noise_str)
        arch = ast.literal_eval(parts[1])
        return noise, arch
    except Exception as e:
        return None, None

def collect_hamming_dynamics(base_dir: Path, target_widths: list, target_noise: float = 0.0):
    """
    Extracts the Inter and Intra Hamming distances at the FINAL LAYER 
    tracked across all EPOCHS to show learning dynamics.
    """
    plot_data = {}
    seeds = [101, 102, 103, 104, 105]
    
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
            
        noise, arch = parse_folder_name(folder.name)
        
        if noise != target_noise or len(arch) == 0 or arch[0] not in target_widths:
            continue
            
        width = arch[0]
        depth = len(arch)
        all_dfs = []
        
        for seed in seeds:
            results_csv = folder / f"evaluated_metrics_seed_{seed}.csv"
            
            if not results_csv.exists():
                continue
                
            df = pd.read_csv(results_csv)
            if df.empty:
                continue
                
            # Filter for the final hidden layer (which holds the ultimate topological representation)
            final_layer_idx = df['layer_idx'].max()
            df_final_layer = df[df['layer_idx'] == final_layer_idx].copy()
            
            all_dfs.append(df_final_layer)
                
        if not all_dfs:
            continue
            
        # Combine and aggregate statistics (mean, std) across all seeds
        combined_df = pd.concat(all_dfs)
        
        # Group by EPOCH this time
        agg_df = combined_df.groupby('epoch').agg({
            'avg_intra_hamming': ['mean', 'std'],
            'avg_inter_hamming': ['mean', 'std']
        }).reset_index()
        
        agg_df.columns = ['epoch', 'intra_mean', 'intra_std', 'inter_mean', 'inter_std']
        
        if width not in plot_data:
            plot_data[width] = {}
            
        plot_data[width][depth] = {
            'epochs': agg_df['epoch'].values,
            'intra_mean': agg_df['intra_mean'].fillna(0).values,
            'intra_std': agg_df['intra_std'].fillna(0).values,
            'inter_mean': agg_df['inter_mean'].fillna(0).values,
            'inter_std': agg_df['inter_std'].fillna(0).values,
            'arch': arch
        }
        
    return plot_data

def plot_topological_dynamics(base_dir_str: str):
    base_dir = Path(base_dir_str)
    
    target_widths = [5, 7, 9, 25]
    
    plot_data = collect_hamming_dynamics(base_dir, target_widths=target_widths, target_noise=0.0)
    
    if not plot_data:
        print("No valid CSV data found.")
        return

    widths = sorted(plot_data.keys())
    depths = sorted({depth for w in widths for depth in plot_data[w].keys()})
    
    fig, axes = plt.subplots(len(widths), len(depths), figsize=(15, 4 * len(widths)), sharey=True, sharex=True)
    
    if len(widths) == 1: axes = [axes]
    if len(depths) == 1: axes = [[ax] for ax in axes]
    
    handles, labels = None, None
    
    for i, width in enumerate(widths):
        for j, depth in enumerate(depths):
            ax = axes[i][j]
            
            if depth not in plot_data[width]:
                ax.axis('off')
                continue
                
            data = plot_data[width][depth]
            epochs = data['epochs']
            
            # --- Plot Intra-class Distance (Cohesion) ---
            intra_m = data['intra_mean']
            intra_s = data['intra_std']
            ax.plot(epochs, intra_m, label='Intra-class Distance (Cohesion)', color='blue', linewidth=2)
            ax.fill_between(epochs, np.maximum(0, intra_m - intra_s), intra_m + intra_s, color='blue', alpha=0.2)
            
            # --- Plot Inter-class Distance (Margin) ---
            inter_m = data['inter_mean']
            inter_s = data['inter_std']
            ax.plot(epochs, inter_m, label='Inter-class Distance (Margin)', color='red', linewidth=2)
            ax.fill_between(epochs, np.maximum(0, inter_m - inter_s), inter_m + inter_s, color='red', alpha=0.2)
            
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            
            # --- Formatting ---
            if i == 0:
                ax.set_title(f"Depth = {depth} Layers", fontsize=14)
            if j == 0:
                ax.set_ylabel(f"Width = {width}\nHamming Distance (Bits)", fontsize=12)
            if i == len(widths) - 1:
                ax.set_xlabel("Training Epochs", fontsize=12)
                
            ax.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle("Training Dynamics: Topological Separation at the Final Layer Over Time", fontsize=18, y=0.98)
    
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                   ncol=2, fontsize=12, frameon=False)
    
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save output
    out_path = neurips_figpath / "hamming_topological_dynamics"
    savefig(fig, str(out_path))

if __name__ == "__main__":
    from src_experiment.paths import outputs
    outputs_dir = outputs / "composite_label_noise"
    
    plot_topological_dynamics(str(outputs_dir))