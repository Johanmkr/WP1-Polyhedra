"""Experiment 1 — routing-information training dynamics.

For each dataset, plots Ĩ(Y;Ω), Ĩ_func(Y;Ω_func) and ρ_func at the deepest layer
across training epochs, ε = 10, mean ± std over seeds, faceted by the relevant
axis: noise level for composite/wbc, PCA target_dim for mnist.

Output: figures/training_dynamics/<dataset>_<arch>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "figures" / "training_dynamics"
EPS = 10.0


def slice_for_arch(df: pd.DataFrame, arch: str) -> pd.DataFrame:
    sub = df[df["arch_str"] == arch]
    if sub.empty:
        return sub
    deepest = sub["layer"].max()
    return sub[(sub["layer"] == deepest) & (np.isclose(sub["epsilon"], EPS))]


def agg(sub: pd.DataFrame, facet_col: str) -> pd.DataFrame:
    """Mean ± std per (epoch, facet) over seeds."""
    g = sub.groupby([facet_col, "epoch"])
    out = g.agg(
        i_raw_mean=("miller_madow_bits", "mean"),
        i_raw_std=("miller_madow_bits", "std"),
        i_func_mean=("miller_madow_func_bits", "mean"),
        i_func_std=("miller_madow_func_bits", "std"),
        rho_func_mean=("rho_func", "mean"),
        rho_mean=("rho", "mean"),
        test_acc_mean=("test_acc", "mean"),
    ).reset_index()
    return out


def plot_panel(ax, agg_df, facet_col, ycol_mean, ycol_std, ylabel, palette):
    facets = sorted(agg_df[facet_col].unique())
    for f, color in zip(facets, palette):
        sub = agg_df[agg_df[facet_col] == f].sort_values("epoch")
        if ycol_std is not None and ycol_std in sub.columns:
            ax.fill_between(sub["epoch"],
                            sub[ycol_mean] - sub[ycol_std],
                            sub[ycol_mean] + sub[ycol_std],
                            alpha=0.15, color=color)
        label = f"{facet_col}={f}"
        ax.plot(sub["epoch"], sub[ycol_mean], "-o", ms=3, color=color, label=label)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def make_figure(df: pd.DataFrame, arch: str, facet_col: str, dataset_label: str,
                outpath: Path) -> bool:
    sub = slice_for_arch(df, arch)
    if sub.empty:
        print(f"  no rows for arch={arch}")
        return False
    H_Y = float(sub["H_Y_bits"].iloc[0])
    a = agg(sub, facet_col)

    facets = sorted(a[facet_col].unique())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(facets), 2)))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    plot_panel(axes[0], a, facet_col, "i_raw_mean", "i_raw_std",
               "Ĩ_raw(Y;Ω) [bits]", cmap)
    plot_panel(axes[1], a, facet_col, "i_func_mean", "i_func_std",
               "Ĩ_func(Y;Ω_func) [bits]", cmap)
    plot_panel(axes[2], a, facet_col, "rho_func_mean", None,
               "ρ_func = |Ω_func|/|Ω|", cmap)
    plot_panel(axes[3], a, facet_col, "test_acc_mean", None,
               "test accuracy", cmap)

    axes[0].axhline(H_Y, ls="--", color="k", lw=0.8, label="H(Y)")
    axes[1].axhline(H_Y, ls="--", color="k", lw=0.8, label="H(Y)")

    deepest = int(sub["layer"].iloc[0])
    fig.suptitle(
        f"{dataset_label} — arch {arch}, layer {deepest}, ε = {EPS}, "
        f"H(Y) = {H_Y:.2f} bits"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"  wrote {outpath}")
    return True


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("\n=== composite ===")
    df = pd.read_csv(REPO / "results" / "composite_label_noise_new_estimator.csv")
    for arch in ["[7, 7, 7]", "[25, 25, 25]", "[5, 5, 5]", "[5, 5, 5, 5, 5]"]:
        make_figure(df, arch, "noise_level", "composite",
                    OUTDIR / f"composite_{arch.replace(', ', '_').replace('[', '').replace(']', '')}.png")

    print("\n=== wbc ===")
    df = pd.read_csv(REPO / "results" / "wbc_label_noise_new_estimator.csv")
    for arch in ["[7, 7, 7]", "[25, 25, 25]", "[5, 5, 5]", "[5, 5, 5, 5, 5]"]:
        make_figure(df, arch, "noise_level", "wbc",
                    OUTDIR / f"wbc_{arch.replace(', ', '_').replace('[', '').replace(']', '')}.png")

    print("\n=== mnist (capacity) ===")
    df = pd.read_csv(REPO / "results" / "mnist_capacity_new_estimator.csv")
    for arch in ["[5, 5, 5]", "[7, 7, 7]"]:
        make_figure(df, arch, "target_dim", "mnist",
                    OUTDIR / f"mnist_{arch.replace(', ', '_').replace('[', '').replace(']', '')}.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
