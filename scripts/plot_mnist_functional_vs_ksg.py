"""Experiment D — Functional MI vs raw MM on MNIST: does the new definition close the gap?

x-axis: PCA dimension.
Three lines: tilde_I_raw (raw MM), tilde_I_func (functional MM, optimal ε),
             KSG k=3 baseline.

Expected: raw and functional track each other; both sit consistently below KSG.

Inputs:
    results/mnist_capacity_new_estimator.csv
    results/mnist_fc_baselines.csv

Outputs:
    figures/mnist_functional_vs_ksg.png
    figures/mnist_functional_vs_ksg.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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

LAST_EPOCH = 150
LAST_LAYER = 3  # 3-layer FC networks
# Use the epsilon that maximises tilde_I_func (typically 10.0 based on existing runs)
EPS_FUNC = 10.0

ARCHS_MNIST = ["[5, 5, 5]", "[7, 7, 7]"]

H_Y_MNIST = 3.319


def load_routing(eps: float) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "mnist_capacity_new_estimator.csv")
    return df[
        (df["epoch"] == LAST_EPOCH)
        & (df["layer"] == LAST_LAYER)
        & (df["arch_str"].isin(ARCHS_MNIST))
        & (np.isclose(df["epsilon"], eps))
    ].copy()


def load_baselines() -> pd.DataFrame:
    path = RESULTS_DIR / "mnist_fc_baselines.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[
        (df["arch_str"].isin(ARCHS_MNIST))
        & (df["layer"] == LAST_LAYER)
    ].copy()


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    routing = load_routing(EPS_FUNC)
    baselines = load_baselines()

    fig, ax = plt.subplots(figsize=(6, 4))

    if not routing.empty:
        for col, label, colour, ls, marker in [
            ("miller_madow_bits",      r"$\tilde{I}_{\mathrm{raw}}$ (routing, M-M)",  "tab:blue",   "-",  "o"),
            ("miller_madow_func_bits", r"$\tilde{I}_{\mathrm{func}}$ (functional, M-M)", "tab:green", "--", "s"),
        ]:
            agg = routing.groupby("target_dim")[col].agg(["mean", "std"]).reset_index()
            xs = agg["target_dim"].to_numpy()
            ys = agg["mean"].to_numpy()
            es = agg["std"].fillna(0).to_numpy()
            ax.plot(xs, ys, color=colour, lw=2, linestyle=ls, marker=marker,
                    markersize=5, label=label)
            ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.12, lw=0)

    if not baselines.empty and "bits_ksg_k3" in baselines.columns:
        agg_ksg = baselines.groupby("target_dim")["bits_ksg_k3"].agg(["mean", "std"]).reset_index()
        xs_k = agg_ksg["target_dim"].to_numpy()
        ys_k = agg_ksg["mean"].to_numpy()
        es_k = agg_ksg["std"].fillna(0).to_numpy()
        ax.plot(xs_k, ys_k, color="tab:cyan", lw=1.5, linestyle="-.", marker="^",
                markersize=5, label=r"KSG $k{=}3$")
        ax.fill_between(xs_k, ys_k - es_k, ys_k + es_k, color="tab:cyan", alpha=0.10, lw=0)

    ax.axhline(H_Y_MNIST, color="k", linestyle="--", lw=0.8, alpha=0.5, label="H(Y)")
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_ylim(bottom=-0.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title(
        rf"MNIST: functional vs raw routing MI (last layer, $\varepsilon={EPS_FUNC:g}$, epoch {LAST_EPOCH})"
    )
    fig.tight_layout()
    savefig(fig, neurips_figpath / "mnist_functional_vs_ksg")


if __name__ == "__main__":
    main()
