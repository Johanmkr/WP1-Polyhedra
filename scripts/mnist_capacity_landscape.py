"""Capacity-vs-bottleneck landscape plots for the MNIST capacity sweep.

Reads results/mnist_capacity_new_estimator.csv and produces:
- heatmaps of I_raw, I_func, and the I_func/I_raw ratio over (target_dim, arch)
  at the deepest layer, ε = 10, last epoch, mean over seeds
- a max-ρ heatmap (trustworthiness)
Also prints a console summary table mirroring the composite sanity script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = Path("results/mnist_capacity_new_estimator.csv")
EPS = 10.0
OUTDIR = Path("figures/mnist_capacity_new_estimator")


def width_from_arch(arch: str) -> int:
    """`[15, 15, 15]` -> 15."""
    inside = arch.strip()[1:-1]
    return int(inside.split(",")[0].strip())


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}; run --aggregate first")
        return 1
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    print(f"loaded {CSV}: shape={df.shape}")

    df["width"] = df["arch_str"].apply(width_from_arch)
    last_epoch = df["epoch"].max()
    print(f"last epoch: {last_epoch}")

    # --- DPI sanity (global) ---
    dpi_ok = (df["plug_in_func_bits"] <= df["plug_in_bits"] + 1e-9).all()
    n_dpi = (~(df["plug_in_func_bits"] <= df["plug_in_bits"] + 1e-9)).sum()
    print(f"[DPI] plug_in_func ≤ plug_in everywhere: {dpi_ok} ({n_dpi} violations)")

    # --- max ρ per cell ---
    rho_max = df.groupby(["target_dim", "width"])["rho"].max().unstack().sort_index().sort_index(axis=1)
    print("\n[trustworthiness] max ρ per cell:")
    print(rho_max.round(3).to_string())

    # --- headline cell: deepest layer, ε=10, last epoch, mean over seeds ---
    sub = df[(df["epoch"] == last_epoch) & (np.isclose(df["epsilon"], EPS))]
    deepest_per_cell = sub.groupby(["target_dim", "width"])["layer"].max().reset_index()
    deepest_per_cell.columns = ["target_dim", "width", "deepest_layer"]
    sub2 = sub.merge(deepest_per_cell, on=["target_dim", "width"])
    sub2 = sub2[sub2["layer"] == sub2["deepest_layer"]]

    head = sub2.groupby(["target_dim", "width"]).agg(
        i_raw=("miller_madow_bits", "mean"),
        i_func=("miller_madow_func_bits", "mean"),
        rho=("rho", "mean"),
        n_regions=("num_regions", "mean"),
        n_quotient=("num_quotient", "mean"),
    ).reset_index()
    head["ratio"] = head["i_func"] / head["i_raw"].replace(0, np.nan)

    print("\n[headline] deepest-layer mean over seeds, ε=10, last epoch:")
    print(head.round(3).to_string(index=False))

    # --- heatmaps ---
    pca_order = sorted(df["target_dim"].unique())
    width_order = sorted(df["width"].unique())

    def pivot(col: str) -> pd.DataFrame:
        return head.pivot(index="target_dim", columns="width", values=col).reindex(
            index=pca_order, columns=width_order
        )

    panels = [
        ("i_raw", "I_raw  [bits]  (Miller-Madow)", "viridis"),
        ("i_func", "I_func  [bits]  (Miller-Madow)", "viridis"),
        ("ratio", "I_func / I_raw", "magma"),
        ("rho", "ρ = R / N", "Reds"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (col, title, cmap) in zip(axes, panels):
        m = pivot(col).values
        im = ax.imshow(m, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(width_order)))
        ax.set_xticklabels([f"[{w}]³" for w in width_order])
        ax.set_yticks(range(len(pca_order)))
        ax.set_yticklabels([f"PCA={d}" for d in pca_order])
        ax.set_title(title)
        ax.set_xlabel("width")
        if ax is axes[0]:
            ax.set_ylabel("PCA target_dim (info bottleneck)")
        # annotate
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "·",
                        ha="center", va="center",
                        color="white" if np.isfinite(v) and v > np.nanmean(m) else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(
        f"MNIST capacity sweep — capacity (width) × information bottleneck (PCA dim)\n"
        f"deepest layer, ε = {EPS}, last epoch (mean over seeds)"
    )
    fig.tight_layout()
    out = OUTDIR / "capacity_landscape.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")

    # --- max ρ heatmap (across all rows, not just headline cell) ---
    fig2, ax = plt.subplots(figsize=(5, 4))
    rho_mat = rho_max.reindex(index=pca_order, columns=width_order).values
    im = ax.imshow(rho_mat, aspect="auto", cmap="Reds", origin="lower",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(width_order)))
    ax.set_xticklabels([f"[{w}]³" for w in width_order])
    ax.set_yticks(range(len(pca_order)))
    ax.set_yticklabels([f"PCA={d}" for d in pca_order])
    ax.set_title("max ρ across (epoch, layer, seed)")
    for i in range(rho_mat.shape[0]):
        for j in range(rho_mat.shape[1]):
            v = rho_mat[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "·",
                    ha="center", va="center", color=color, fontsize=9)
    fig2.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig2.tight_layout()
    out2 = OUTDIR / "trustworthiness_max_rho.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
