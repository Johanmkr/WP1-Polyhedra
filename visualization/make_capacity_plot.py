import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src_experiment.paths import neurips_figpath
    from src_experiment.utils import savefig
except ImportError:
    print("❌ Error: Could not import 'geobin_py.reconstruction'.")
    sys.exit(1)
    

# --- 1. CONFIGURATION & STYLING ---
sns.set_theme(style="white", context="paper", font_scale=1.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL DEFINITION ---
class TrackingMLP(nn.Module):
    """An MLP that captures binary activation patterns across ALL hidden layers."""
    def __init__(self, d_in, d_n, L):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_n)])
        for _ in range(L - 1):
            self.layers.append(nn.Linear(d_n, d_n))
            
    def forward(self, x):
        patterns = []
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # Binary pattern: 1 for active, 0 for inactive
            patterns.append((out > 0).int())
            if i < len(self.layers) - 1:
                out = torch.relu(out)
        return torch.cat(patterns, dim=1)

# --- 3. METRICS ---
def get_smoothed_mi(N, dx, dn, L, n_reps=2):
    """Calculates Normalized MI, averaged over multiple random initializations."""
    mi_estimates = []
    
    for _ in range(n_reps):
        X = torch.randn(N, dx, device=device)
        model = TrackingMLP(dx, dn, L).to(device)
        
        with torch.no_grad():
            T = model(X).cpu().numpy()
            
        _, counts = np.unique(T, axis=0, return_counts=True)
        probs = counts / np.sum(counts)
        H_T = entropy(probs, base=2)
        H_X = np.log2(N)
        
        mi_estimates.append(H_T / H_X if H_X > 0 else 0.0)
        
    return np.mean(mi_estimates)

# --- 4. EXPERIMENT ROUTINE ---
def run_grid_experiment(dx, L, ax):
    # Setup grid resolutions
    N_max, n_max = 5000, 40
    
    # Logarithmic spacing for N to capture exponential capacity curves
    # Using np.unique ensures strictly increasing integer values
    N_vals = np.unique(np.logspace(np.log10(10), np.log10(N_max), 50).astype(int))
    dn_vals = np.linspace(2, n_max, 20).astype(int)
    
    mi_matrix = np.zeros((len(dn_vals), len(N_vals)))
    print(f"Running grid: d_x={dx}, L={L} (Points: {len(N_vals)}x{len(dn_vals)})...")
    
    for i, dn in enumerate(dn_vals):
        for j, N in enumerate(N_vals):
            mi_matrix[i, j] = get_smoothed_mi(N, dx, dn, L)
            
    X_mesh, Y_mesh = np.meshgrid(N_vals, dn_vals)
    
    cmap = plt.colormaps['Spectral_r'].copy()
    cmap.set_over('white')
    
    # shading='nearest' works best for non-uniform/logarithmic coordinates
    cp = ax.pcolormesh(X_mesh, Y_mesh, mi_matrix, shading='nearest', cmap=cmap, vmin=0, vmax=0.99)
    
    # --- CALCULATE THEORETICAL BOUNDARIES ---
    n_line = np.linspace(2, n_max, 100)
    bound_max = []
    bound_exp = []
    
    for n in n_line:
        # Schläfli Sum
        base_sum = sum(math.comb(int(n), i) for i in range(min(int(n) + 1, dx + 1)))
        
        # Montúfar Multiplier
        multiplier = (max(1, math.floor(int(n) / dx)) ** dx) ** (L - 1) if L > 1 else 1
        bound_max.append(multiplier * base_sum)
        
        # Hanin-Rolnick Expected Bound
        N_total = L * int(n)
        bound_exp.append(sum(math.comb(N_total, i) for i in range(min(N_total + 1, dx + 1))))
        
    # Plot boundaries
    label_max = r'$R_{max}$'
    ax.plot(bound_max, n_line, color='black', lw=2, ls='--', label=label_max)
    ax.plot(bound_exp, n_line, color='red', lw=2, ls=':', label=r'$\mathbb{E}[R]$')
    
    # Formatting
    ax.set_xscale('log') # Explicitly set x-axis to log scale
    ax.set_title(f'$d_x={dx}, L={L}$', fontweight='bold', fontsize=12)
    ax.set_xlim(10, N_max)
    ax.set_ylim(2, n_max)
    ax.grid(True, linestyle=':', alpha=0.4, which='both') # grid for both major/minor log ticks
    
    return cp

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # Define our 3x3 test matrix parameters
    dx_values = [2, 10, 50,100]  # Columns: Increasing input dimension
    L_values = [1, 5, 15]     # Rows: Increasing depth
    
    fig, axes = plt.subplots(len(L_values), len(dx_values), figsize=(18, 10), 
                             sharex=True, sharey=True, constrained_layout=True)
    
    cp_final = None
    
    for row_idx, L in enumerate(L_values):
        for col_idx, dx in enumerate(dx_values):
            ax = axes[row_idx, col_idx]
            cp_final = run_grid_experiment(dx=dx, L=L, ax=ax)
            
            # Add legends only to the far-right plots to keep it clean
            if col_idx == len(dx_values) - 1:
                ax.legend(loc='lower right', frameon=True, fontsize=10)

    # --- GLOBAL LABELS & LEGENDS ---
    fig.supxlabel('Number of Samples ($N$) - Log Scale', fontsize=16, fontweight='bold')
    fig.supylabel('Network Width per Layer ($n$)', fontsize=16, fontweight='bold')
    fig.suptitle('Neural Network Capacity: $I(X;T)/H(X)$ across Dimension and Depth', fontsize=20)
    
    cbar = fig.colorbar(cp_final, ax=axes, location='right', aspect=50, shrink=0.9)
    cbar.set_label('Normalized MI (1.0 = Perfect Memorization)', fontsize=14)
    
    # plt.savefig(neurips_figpath / "capacity.pdf", dpi=300)
    savefig(fig, neurips_figpath / "capacity")
    plt.show()