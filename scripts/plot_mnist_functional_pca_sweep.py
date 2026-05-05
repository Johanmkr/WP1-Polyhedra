"""Functional MI vs epoch, sweeping PCA dimension for each epsilon value.

One panel per epsilon; lines within each panel show different PCA dimensions
coloured by viridis.  Illustrates which (epsilon, PCA-dim) combinations keep
tilde_I_func in the informative regime throughout training.

Inputs:
    results/mnist_capacity_new_estimator.csv

Outputs:
    figures/mnist_functional_pca_sweep.png
    figures/mnist_functional_pca_sweep.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src_experiment.utils import savefig
from src_experiment.paths import neurips_figpath

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"

H_Y_MNIST = 3.319  # bits, 10-class uniform

# Use only the deepest hidden layer (layer 3 for 3-layer nets)
TARGET_LAST_LAYER = 3

# Epsilon values to show as separate panels (subset for readability)
EPSILONS_TO_SHOW = [0.01, 0.1, 1.0, 10.0, 100.0]


def load_data() -> pd.DataFrame:
    path = RESULTS_DIR / "mnist_capacity_new_estimator.csv"
    df = pd.read_csv(path)
    df = df[df["layer"] == TARGET_LAST_LAYER].copy()
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["epsilon", "target_dim", "epoch"])["miller_madow_func_bits"]
        .agg(["mean", "std"])
        .reset_index()
    )


def plot(agg: pd.DataFrame) -> None:
    epsilons = [e for e in EPSILONS_TO_SHOW if np.isclose(agg["epsilon"].unique(), e).any()]
    if not epsilons:
        epsilons = sorted(agg["epsilon"].unique())

    n_eps = len(epsilons)
    ncols = min(n_eps, 4)
    nrows = (n_eps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    pca_dims = sorted(agg["target_dim"].unique())
    cmap = cm.viridis
    colours = {d: cmap(i / max(len(pca_dims) - 1, 1)) for i, d in enumerate(pca_dims)}

    for idx, eps in enumerate(epsilons):
        ax = axes[idx // ncols][idx % ncols]
        sub = agg[np.isclose(agg["epsilon"], eps)]
        for d in pca_dims:
            row = sub[sub["target_dim"] == d].sort_values("epoch")
            if row.empty:
                continue
            xs = row["epoch"].to_numpy()
            ys = row["mean"].to_numpy()
            errs = row["std"].fillna(0).to_numpy()
            ax.plot(xs, ys, color=colours[d], lw=1.5, label=f"d={d}")
            ax.fill_between(xs, ys - errs, ys + errs, color=colours[d], alpha=0.15, lw=0)

        ax.axhline(H_Y_MNIST, color="k", linestyle="--", lw=0.9, alpha=0.6, label="H(Y)")
        ax.text(xs[-1] * 0.02, H_Y_MNIST + 0.05, "H(Y)", fontsize=8, alpha=0.7)
        ax.set_ylim(bottom=-0.05, top=H_Y_MNIST + 0.3)
        ax.set_title(rf"$\varepsilon = {eps:g}$")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\tilde{I}_{\mathrm{func}}$ (bits)")
        ax.grid(alpha=0.25)

    # Hide unused axes
    for idx in range(len(epsilons), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared colourbar for PCA dims
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(pca_dims), vmax=max(pca_dims)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("PCA dim")
    cbar.set_ticks(pca_dims)

    fig.suptitle(
        r"MNIST: $\tilde{I}_{\mathrm{func}}$ vs epoch by PCA dimension"
        f" (last hidden layer, {len(pca_dims)} dims)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    savefig(fig, neurips_figpath / "mnist_functional_pca_sweep")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    df = load_data()
    if df.empty:
        raise SystemExit("No data found — run mnist_capacity experiments first.")
    agg = aggregate(df)
    plot(agg)


if __name__ == "__main__":
    main()
