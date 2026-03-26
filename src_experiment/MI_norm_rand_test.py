import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Import your existing evaluator
from estimate_quantities import ExperimentEvaluator

def run_memorization_experiment(true_labels_h5, random_labels_h5):
    print("Evaluating True Labels Network...")
    eval_true = ExperimentEvaluator(true_labels_h5)
    df_true = eval_true.evaluate_all()
    df_true['Condition'] = 'True Labels (Generalization)'
    
    # Calculate H(Y) to normalize I(Y;W)
    _, counts = np.unique(eval_true.labels, return_counts=True)
    H_Y_true = -np.sum((counts / len(eval_true.labels)) * np.log2(counts / len(eval_true.labels)))
    df_true['Normalized Routing Info'] = df_true['I(Y;W)'] / H_Y_true
    
    print("\nEvaluating Random Labels Network...")
    eval_rand = ExperimentEvaluator(random_labels_h5)
    df_rand = eval_rand.evaluate_all()
    df_rand['Condition'] = 'Random Labels (Memorization)'
    
    _, counts = np.unique(eval_rand.labels, return_counts=True)
    H_Y_rand = -np.sum((counts / len(eval_rand.labels)) * np.log2(counts / len(eval_rand.labels)))
    df_rand['Normalized Routing Info'] = df_rand['I(Y;W)'] / H_Y_rand

    # Combine data
    df_combined = pd.concat([df_true, df_rand], ignore_index=True)
    
    # --- PLOTTING ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Assuming you only want to plot the final epoch
    final_epoch = df_combined['epoch'].max()
    plot_data = df_combined[df_combined['epoch'] == final_epoch]
    
    sns.lineplot(
        data=plot_data, 
        x='layer_idx', 
        y='Normalized Routing Info', 
        hue='Condition', 
        marker='o',
        linewidth=2.5,
        markersize=8,
        palette=['#1f77b4', '#d62728'],
        ax=ax
    )
    
    ax.set_title('Routing Information $I(Y;\Omega)$ Across Depth', fontweight='bold', pad=15)
    ax.set_xlabel('Network Layer Index')
    ax.set_ylabel('Region Purity ($I(Y;\Omega) / H(Y)$)')
    ax.set_ylim(0, 1.05)
    
    # Add a horizontal line at 1.0 to denote perfect structural memorization
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Purity limit')
    ax.legend(frameon=True, loc='lower right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Point these to your HDF5 files generated from your training runs
    run_memorization_experiment(
        true_labels_h5="outputs/mnist_mem_gen_exp/mnist_minimal/seed_101.h5", 
        random_labels_h5="outputs/mnist_mem_gen_exp/mnist_minimal_random/seed_101.h5"
    )