import os
import sys
import ast
import h5py
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

def collect_region_accuracy_data(base_dir: Path, target_noise: float = 0.0):
    plot_data = {}
    seeds = [101, 102, 103, 104, 105]
    
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
            
        noise, arch = parse_folder_name(folder.name)
        if noise != target_noise or not arch:
            continue
            
        arch_tuple = tuple(arch)
        all_dfs = []
        
        for seed in seeds:
            h5_path = folder / f"seed_{seed}.h5"
            regions_csv = folder / f"evaluated_metrics_seed_{seed}.csv"
            
            if not h5_path.exists() or not regions_csv.exists():
                continue
                
            # 1. Load the region data from the cached CSV
            df_reg = pd.read_csv(regions_csv)
            if df_reg.empty:
                continue
                
            # Filter for the final layer to get the total network capacity
            final_layer_idx = df_reg['layer_idx'].max()
            df_reg_final = df_reg[df_reg['layer_idx'] == final_layer_idx].copy()
            
            # 2. Extract accuracy directly from the .h5 file
            with h5py.File(h5_path, 'r') as f:
                if 'training_results' in f and 'test_accuracy' in f['training_results']:
                    acc_array = np.array(f['training_results']['test_accuracy'])
                    
                    epochs = np.arange(len(acc_array))
                    df_acc = pd.DataFrame({'epoch': epochs, 'test_accuracy': acc_array})
                else:
                    print(f"⚠️ Warning: 'test_accuracy' not found in {h5_path.name}")
                    continue
            
            # Merge the regions and accuracy DataFrames on the 'epoch' column
            df_merged = pd.merge(df_reg_final, df_acc, on='epoch', how='inner')
            all_dfs.append(df_merged)
                
        if not all_dfs:
            continue
            
        # Combine across seeds and average
        combined_df = pd.concat(all_dfs)
        agg_df = combined_df.groupby('epoch').agg({
            'populated_regions': ['mean', 'std'],
            'test_accuracy': ['mean', 'std']
        }).reset_index()
        
        agg_df.columns = ['epoch', 'pop_mean', 'pop_std', 'acc_mean', 'acc_std']
        plot_data[arch_tuple] = agg_df
        
    return plot_data

def plot_regions_vs_accuracy(base_dir_str: str):
    base_dir = Path(base_dir_str)
    
    # Extract data for clean dataset
    plot_data = collect_region_accuracy_data(base_dir, target_noise=0.0)
    
    if not plot_data:
        print("No valid merged CSV/H5 data found.")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Sort architectures by depth first, then width
    sorted_archs = sorted(plot_data.keys(), key=lambda x: (len(x), x[0]))
    
    colors = {5: 'blue', 7: 'orange', 9: 'green', 25: 'red'}
    markers = {3: 'o', 4: 's', 5: '^'}
    
    for arch in sorted_archs:
        df = plot_data[arch]
        width = arch[0]
        depth = len(arch)
        
        c = colors.get(width, 'gray')
        m = markers.get(depth, '.')
        label = f"Arch: {list(arch)}"
        
        # --- AXES SWITCHED ---
        ax.plot(df['acc_mean'], df['pop_mean'], color=c, alpha=0.5, linewidth=1.5)
        ax.scatter(df['acc_mean'], df['pop_mean'], color=c, marker=m, s=40, label=label, edgecolor='w', linewidth=0.5)
        
        # Annotate the final epoch to show where the trajectory ends
        final_x = df['acc_mean'].iloc[-1]
        final_y = df['pop_mean'].iloc[-1]
        ax.scatter(final_x, final_y, color=c, marker='*', s=150, edgecolor='black', zorder=5)

    # --- Y-AXIS FORMATTING ---
    # ax.set_yscale('log')
    ax.set_ylabel("Populated Regions", fontsize=12)

    # --- X-AXIS FORMATTING (3-Part Custom Piecewise Linear Scale) ---
    # f(0.9) = 0.9
    # f(0.97) = 0.9 + (0.07 * 10) = 1.6
    # f(1.0) = 1.6 + (0.03 * 30) = 2.5
    def forward(x):
        return np.piecewise(x, 
            [x <= 0.9, (x > 0.9) & (x <= 0.97), x > 0.97],
            [lambda x: x, 
             lambda x: 0.9 + (x - 0.9) * 10, 
             lambda x: 1.6 + (x - 0.97) * 30])

    def inverse(y):
        return np.piecewise(y, 
            [y <= 0.9, (y > 0.9) & (y <= 1.6), y > 1.6],
            [lambda y: y, 
             lambda y: 0.9 + (y - 0.9) / 10, 
             lambda y: 0.97 + (y - 1.6) / 30])

    ax.set_xscale('function', functions=(forward, inverse))
    ax.set_xlim(0.2, 1.002) # slightly above 1.0 to fit the final label
    
    # Create the explicit ticks for the 3 zones
    standard_ticks = np.round(np.arange(0.2, 0.9, 0.1), 2)
    dense_ticks = np.round(np.arange(0.9, 0.97, 0.01), 2)
    hyper_dense_ticks = np.round(np.arange(0.97, 1.001, 0.005), 3)
    
    all_ticks = np.concatenate([standard_ticks, dense_ticks, hyper_dense_ticks])
    
    ax.set_xticks(all_ticks)
    
    # Format labels (strip trailing zeros from 0.9x but keep 3 decimals for the high-res end)
    labels = []
    for t in all_ticks:
        if t >= 0.97:
            labels.append(f"{t:.3f}")
        else:
            labels.append(f"{t:.2f}")
            
    ax.set_xticklabels(labels, rotation=45, fontsize=8, ha='right')
    ax.set_xlabel("Test Accuracy", fontsize=12)

    ax.set_title("Geometric Capacity vs. Generalization Trajectories", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legend setup
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Architectures", fontsize=10)
    
    # Custom legend entry for the "Star" (End of training)
    ax.plot([], [], marker='*', color='black', linestyle='None', markersize=10, label='Final Epoch')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    fig.tight_layout()
    
    # Save output
    out_path = neurips_figpath / "regions_vs_accuracy_trajectory"
    savefig(fig, str(out_path))

if __name__ == "__main__":
    from src_experiment.paths import outputs
    outputs_dir = outputs / "composite_label_noise"
    
    plot_regions_vs_accuracy(str(outputs_dir))