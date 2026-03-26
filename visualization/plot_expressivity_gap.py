import os
import sys
import ast
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the project root is in the path to import src_experiment
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src_experiment.estimate_quantities import ExperimentEvaluator
from src_experiment.utils import savefig
from src_experiment.paths import neurips_figpath 

def calculate_montufar_bound(widths: list, input_dim: int = 2) -> int:
    """
    Calculates the maximum number of linear regions for a deep ReLU network 
    based on Montufar et al. (2014).
    """
    if not widths:
        return 1
        
    L = len(widths)
    # 1. Product term over hidden layers up to L-1
    prod_term = 1
    for i in range(L - 1):
        prod_term *= math.floor(widths[i] / input_dim) ** input_dim
        
    # 2. Summation term for the final layer L
    sum_term = 0
    n_L = widths[-1]
    for j in range(min(input_dim, n_L) + 1):
        sum_term += math.comb(n_L, j)
        
    return prod_term * sum_term

def parse_folder_name(folder_name: str):
    """Parses 'n0.0_[5, 5, 5]' into noise=0.0, arch=[5, 5, 5]"""
    try:
        parts = folder_name.split('_', 1)
        noise_str = parts[0].replace('n', '')
        noise = float(noise_str)
        arch = ast.literal_eval(parts[1])
        return noise, arch
    except Exception as e:
        print(f"Skipping malformed folder {folder_name}: {e}")
        return None, None

def collect_data(base_dir: Path, target_widths: list, target_noise: float = 0.0):
    """
    Iterates over outputs, processes the 5 seeds per config, and aggregates
    the mean and standard deviation for the target noise and widths.
    """
    plot_data = {}
    seeds = [101, 102, 103, 104, 105]
    
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
            
        noise, arch = parse_folder_name(folder.name)
        
        # Filter for our constraints
        if noise != target_noise or len(arch) == 0 or arch[0] not in target_widths:
            continue
            
        width = arch[0]
        depth = len(arch)
        all_dfs = []
        
        # Process each seed run
        for seed in seeds:
            h5_path = folder / f"seed_{seed}.h5"
            if not h5_path.exists():
                print(f"  ⚠️ Missing seed file: {h5_path.name} in {folder.name}")
                continue
                
            results_csv = folder / f"evaluated_metrics_seed_{seed}.csv"
            
            # Cache results to prevent re-evaluating the h5 tree repeatedly
            if results_csv.exists():
                df = pd.read_csv(results_csv)
            else:
                print(f"Evaluating {folder.name} / seed_{seed}...")
                evaluator = ExperimentEvaluator(h5_path)
                df = evaluator.evaluate_all()
                df.to_csv(results_csv, index=False)
                
            if not df.empty:
                # Filter for the FINAL layer capacity
                final_layer_idx = max(df['layer_idx'])
                df_final = df[df['layer_idx'] == final_layer_idx].sort_values('epoch')
                all_dfs.append(df_final)
                
        if not all_dfs:
            continue
            
        # Combine and aggregate statistics (mean, std) across all seeds
        combined_df = pd.concat(all_dfs)
        agg_df = combined_df.groupby('epoch').agg({
            'total_regions': ['mean', 'std'],
            'populated_regions': ['mean', 'std']
        }).reset_index()
        
        # Flatten MultiIndex columns
        agg_df.columns = ['epoch', 'total_mean', 'total_std', 'pop_mean', 'pop_std']
        
        r_max = calculate_montufar_bound(arch, input_dim=2)
        
        if width not in plot_data:
            plot_data[width] = {}
            
        plot_data[width][depth] = {
            'epochs': agg_df['epoch'].values,
            'total_mean': agg_df['total_mean'].fillna(0).values,
            'total_std': agg_df['total_std'].fillna(0).values,
            'pop_mean': agg_df['pop_mean'].fillna(0).values,
            'pop_std': agg_df['pop_std'].fillna(0).values,
            'r_max': r_max,
            'arch': arch
        }
        
    return plot_data

def plot_expressivity_gap(base_dir_str: str):
    base_dir = Path(base_dir_str)
    target_widths = [5, 7, 9, 25]
    
    plot_data = collect_data(base_dir, target_widths=target_widths, target_noise=0.0)
    
    if not plot_data:
        print("No valid data found to plot.")
        return

    widths = sorted(plot_data.keys())
    depths = sorted({depth for w in widths for depth in plot_data[w].keys()})
    
    fig, axes = plt.subplots(len(widths), len(depths), figsize=(15, 4 * len(widths)), sharex=True, sharey=True)
    
    # Handle grid edge cases
    if len(widths) == 1: axes = [axes]
    if len(depths) == 1: axes = [[ax] for ax in axes]
    
    # Variables to store legend handles
    handles, labels = None, None
    
    for i, width in enumerate(widths):
        for j, depth in enumerate(depths):
            ax = axes[i][j]
            
            if depth not in plot_data[width]:
                ax.axis('off')
                continue
                
            data = plot_data[width][depth]
            epochs = data['epochs']
            
            # --- Plot Total Regions ---
            t_mean = data['total_mean']
            t_std = data['total_std']
            ax.plot(epochs, t_mean, label='Total Regions (Empirical)', color='blue', linewidth=2)
            ax.fill_between(epochs, np.maximum(1, t_mean - t_std), t_mean + t_std, color='blue', alpha=0.2)
            
            # --- Plot Populated Regions ---
            p_mean = data['pop_mean']
            p_std = data['pop_std']
            ax.plot(epochs, p_mean, label='Populated Regions (Data)', color='orange', linewidth=2)
            ax.fill_between(epochs, np.maximum(1, p_mean - p_std), p_mean + p_std, color='orange', alpha=0.2)
            
            # --- Plot R_max ---
            ax.axhline(data['r_max'], color='red', linestyle='--', label=r'Theoretical $R_{max}$')
            
            # --- Capture handles for the figure legend ---
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            
            # --- Formatting ---
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(f"Depth = {depth} Layers", fontsize=14)
            if j == 0:
                ax.set_ylabel(f"Width = {width}\nLog(Regions)", fontsize=12)
            if i == len(widths) - 1:
                ax.set_xlabel("Epochs", fontsize=12)
                
            ax.grid(True, linestyle=':', alpha=0.6)

    # --- Add Figure-Level Legend and Title ---
    fig.suptitle("The Expressivity Gap (Clean Data): Theoretical vs. Empirical Capacity", fontsize=18, y=0.98)
    
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                   ncol=3, fontsize=12, frameon=False)
    
    # Adjust layout so subplots don't overlap with the title and legend at the top
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save output using the imported neurips_figpath
    out_path = neurips_figpath / "expressivity_gap_architecture"
    savefig(fig, str(out_path))

if __name__ == "__main__":
    # Ensure this matches your actual outputs path
    current_dir = Path(__file__).resolve().parent.parent
    outputs_dir = current_dir / "outputs" / "composite_label_noise"
    
    plot_expressivity_gap(str(outputs_dir))