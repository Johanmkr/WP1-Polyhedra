import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src_experiment.paths import neurips_figpath
    from src_experiment.utils import savefig
except ImportError:
    print("⚠️ Warning: Could not import custom paths/utils. Using standard plotting.")
    neurips_figpath = Path("./")
    def savefig(fig, path): fig.savefig(f"{path}.pdf", dpi=300)

# --- 1. CONFIGURATION & STYLING ---
sns.set_theme(style="white", context="paper", font_scale=1.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL DEFINITION ---
class TrackingMLP(nn.Module):
    """An MLP that captures discrete routing patterns (\Omega) across ALL hidden layers."""
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
            # Binary structural pattern: 1 for active, 0 for dead ReLU
            patterns.append((out > 0).int())
            if i < len(self.layers) - 1:
                out = torch.relu(out)
        return torch.cat(patterns, dim=1)

# --- 3. METRICS (SHATTERING CAPACITY) ---
def get_shattering_capacity(N, dx, dn, L, n_classes=2, n_reps=2):
    """
    Calculates the raw empirical Routing Information I(Y;\Omega),
    normalized by H(Y). This directly measures the network's ability 
    to isolate random samples into pure, single-class regions.
    """
    capacity_estimates = []
    
    for _ in range(n_reps):
        # 1. Random Data and Random Labels (Pure Memorization Task)
        X = torch.randn(N, dx, device=device)
        Y = torch.randint(0, n_classes, (N,), device=device)
        
        model = TrackingMLP(dx, dn, L).to(device)
        
        with torch.no_grad():
            raw_patterns = model(X).cpu().numpy()
            
        # 2. Hash binary patterns to unique categorical region IDs (\Omega)
        _, Omega = np.unique(raw_patterns, axis=0, return_inverse=True)
        Y_np = Y.cpu().numpy()
        
        # 3. Compute empirical I(Y;\Omega) in Bits (NO BIAS CORRECTION)
        mi_nats = mutual_info_score(Y_np, Omega)
        mi_bits = mi_nats / np.log(2)
        
        # 4. Normalize by Source Information H(Y)
        _, counts = np.unique(Y_np, return_counts=True)
        H_Y = entropy(counts, base=2)
        
        # If H(Y) is 0, capacity is undefined/0. Otherwise, normalize to [0, 1]
        capacity = (mi_bits / H_Y) if H_Y > 0 else 0.0
        capacity_estimates.append(capacity)
        
    return np.mean(capacity_estimates)

# --- 4. EXPERIMENT ROUTINE ---
def run_grid_experiment(dx, L, ax):
    # Setup grid resolutions
    N_max, n_max = 5000, 40
    
    N_vals = np.unique(np.logspace(np.log10(10), np.log10(N_max), 50).astype(int))
    dn_vals = np.linspace(2, n_max, 20).astype(int)
    
    capacity_matrix = np.zeros((len(dn_vals), len(N_vals)))
    print(f"Running grid: d_x={dx}, L={L} (Points: {len(N_vals)}x{len(dn_vals)})...")
    
    for i, dn in enumerate(dn_vals):
        for j, N in enumerate(N_vals):
            # Evaluate Shattering Capacity
            capacity_matrix[i, j] = get_shattering_capacity(N, dx, dn, L, n_classes=2)
            
    X_mesh, Y_mesh = np.meshgrid(N_vals, dn_vals)
    
    # 'magma' or 'inferno' are excellent for showing phase transitions
    cmap = plt.colormaps['magma'].copy()
    
    cp = ax.pcolormesh(X_mesh, Y_mesh, capacity_matrix, shading='nearest', cmap=cmap, vmin=0, vmax=1.0)
    
    # --- CALCULATE THEORETICAL BOUNDARY (R_max) ---
    n_line = np.linspace(2, n_max, 100)
    bound_max = []
    
    for n in n_line:
        # Schläfli Sum (Base Capacity for a single layer)
        base_sum = sum(math.comb(int(n), i) for i in range(min(int(n) + 1, dx + 1)))
        
        # Montúfar Multiplier (Exponential folding due to depth)
        multiplier = (max(1, math.floor(int(n) / dx)) ** dx) ** (L - 1) if L > 1 else 1
        bound_max.append(multiplier * base_sum)
        
    ax.plot(bound_max, n_line, color='cyan', lw=2, ls='--', label=r'$R_{max}$')
    
    ax.set_xscale('log')
    ax.set_title(f'$d_x={dx}, L={L}$', fontweight='bold', fontsize=12)
    ax.set_xlim(10, N_max)
    ax.set_ylim(2, n_max)
    ax.grid(True, linestyle=':', alpha=0.3, which='both')
    
    return cp

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    dx_values = [2, 10, 50, 100]
    L_values = [1, 5, 15] 
    
    fig, axes = plt.subplots(len(L_values), len(dx_values), figsize=(18, 10), 
                             sharex=True, sharey=True, constrained_layout=True)
    
    cp_final = None
    
    for row_idx, L in enumerate(L_values):
        for col_idx, dx in enumerate(dx_values):
            ax = axes[row_idx, col_idx]
            cp_final = run_grid_experiment(dx=dx, L=L, ax=ax)
            
            if col_idx == len(dx_values) - 1:
                # White text legend to contrast with dark magma background
                legend = ax.legend(loc='lower right', frameon=True, fontsize=10)
                legend.get_frame().set_alpha(0.8)

    fig.supxlabel('Number of Samples ($N$) - Log Scale', fontsize=16, fontweight='bold')
    fig.supylabel('Network Width per Layer ($n$)', fontsize=16, fontweight='bold')
    # fig.suptitle('Structural Shattering Capacity: Region Purity across Dimension and Depth', fontsize=20)
    
    cbar = fig.colorbar(cp_final, ax=axes, location='right', aspect=50, shrink=0.9)
    cbar.set_label('Region Purity ($I(Y;\Omega) / H(Y)$)', fontsize=14)
    
    savefig(fig, neurips_figpath / "shattering_capacity")
    plt.show()