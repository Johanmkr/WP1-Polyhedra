"""Experiment 2 — ρ_func ↔ generalization gap.

For each dataset, slices the aggregated estimator CSV to the deepest layer,
ε = 10, and the last available epoch per (arch, facet, seed). Computes a
noise-adjusted generalization gap

    gen_gap_norm = train_acc − (1 − noise) · test_acc

(reduces to plain `train − test` for the no-noise mnist run) and a simple
absolute gap |train − test|. Aggregates mean over seeds per cell, then
plots a 2-row × 3-column figure (rows: gen_gap_norm, |train−test|;
columns: composite, wbc, mnist) of ρ_func vs gap, colored by arch.

Reports Pearson and Spearman correlation with 95% bootstrap CI per
(dataset, gap-metric) cell. Trustworthy-only correlations (mean ρ ≤ 0.3)
are reported alongside the full ones.

Outputs:
- figures/rho_func_vs_gen_gap.png
- results/rho_func_vs_gen_gap_correlations.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
EPS = 10.0
RHO_TRUST = 0.3
RNG = np.random.default_rng(20260427)
N_BOOT = 2000

DATASETS = [
    ("composite", "composite_label_noise_new_estimator.csv", "noise_level"),
    ("wbc",       "wbc_label_noise_new_estimator.csv",       "noise_level"),
    ("mnist",     "mnist_capacity_new_estimator.csv",        "target_dim"),
]


def deepest_last_epoch_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Per (arch_str, seed), keep only the deepest layer at ε = EPS at
    the largest epoch present for that (arch, seed) row group."""
    deepest = df.groupby("arch_str")["layer"].transform("max")
    sub = df[(df["layer"] == deepest) & (np.isclose(df["epsilon"], EPS))].copy()
    last_ep = sub.groupby(["arch_str", "seed"])["epoch"].transform("max")
    return sub[sub["epoch"] == last_ep].copy()


def add_gaps(df: pd.DataFrame, facet_col: str) -> pd.DataFrame:
    if facet_col == "noise_level":
        n = df["noise_level"].to_numpy()
    else:
        n = np.zeros(len(df))
    df = df.assign(
        gen_gap_norm=df["train_acc"] - (1.0 - n) * df["test_acc"],
        gen_gap_abs=(df["train_acc"] - df["test_acc"]).abs(),
    )
    return df


def aggregate(df: pd.DataFrame, facet_col: str) -> pd.DataFrame:
    """Mean over seeds per (arch_str, facet) cell."""
    g = df.groupby(["arch_str", facet_col])
    out = g.agg(
        rho_func=("rho_func", "mean"),
        rho=("rho", "mean"),
        gen_gap_norm=("gen_gap_norm", "mean"),
        gen_gap_abs=("gen_gap_abs", "mean"),
        train_acc=("train_acc", "mean"),
        test_acc=("test_acc", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return out


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n: int = N_BOOT) -> dict:
    if len(x) < 3:
        return {"pearson": np.nan, "pearson_lo": np.nan, "pearson_hi": np.nan,
                "spearman": np.nan, "spearman_lo": np.nan, "spearman_hi": np.nan,
                "n_cells": len(x)}
    p_obs = pearsonr(x, y).statistic
    s_obs = spearmanr(x, y).statistic
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    ps = np.empty(n); ss = np.empty(n)
    for i in range(n):
        xi, yi = x[idx[i]], y[idx[i]]
        if np.std(xi) == 0 or np.std(yi) == 0:
            ps[i] = np.nan; ss[i] = np.nan; continue
        ps[i] = pearsonr(xi, yi).statistic
        ss[i] = spearmanr(xi, yi).statistic
    return {
        "pearson": p_obs,
        "pearson_lo": float(np.nanpercentile(ps, 2.5)),
        "pearson_hi": float(np.nanpercentile(ps, 97.5)),
        "spearman": s_obs,
        "spearman_lo": float(np.nanpercentile(ss, 2.5)),
        "spearman_hi": float(np.nanpercentile(ss, 97.5)),
        "n_cells": len(x),
    }


def parse_arch(arch_str: str) -> tuple[int, int]:
    """Return (depth, width) for a uniform arch string like '[7, 7, 7]'."""
    nums = [int(s) for s in arch_str.strip("[] ").split(",")]
    return len(nums), nums[0]


def make_palette(archs: list[str]) -> dict[str, tuple]:
    """Color by depth (depth ∈ {3,4,5} → distinct hues)."""
    depth_colors = {3: "#1f77b4", 4: "#2ca02c", 5: "#d62728"}
    palette = {}
    widths_by_depth: dict[int, list[int]] = {}
    for a in archs:
        d, w = parse_arch(a)
        widths_by_depth.setdefault(d, []).append(w)
    for a in archs:
        d, w = parse_arch(a)
        widths = sorted(set(widths_by_depth[d]))
        # lighten/darken by width: lighter = smaller width
        idx = widths.index(w)
        n = len(widths)
        base = plt.matplotlib.colors.to_rgb(depth_colors.get(d, "#666666"))
        # blend to white for smaller widths
        t = 0.2 + 0.6 * (idx / max(n - 1, 1))
        rgb = tuple(t * c + (1 - t) * 1.0 for c in base)
        palette[a] = rgb
    return palette


def plot_scatter(ax, agg_df, facet_col, ymetric, ylabel, palette,
                 title, facet_marker_map, show_xlabel: bool):
    for arch, g in agg_df.groupby("arch_str"):
        for facet_val, gg in g.groupby(facet_col):
            marker = facet_marker_map[facet_val]
            ax.scatter(gg["rho_func"], gg[ymetric],
                       color=palette[arch], marker=marker, s=80,
                       edgecolor="black", linewidth=0.5)
    if show_xlabel:
        ax.set_xlabel(r"$\rho_{\mathrm{func}}$ (deepest, ε=10, last epoch)")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", lw=0.6, ls=":", alpha=0.5)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)


def main() -> int:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5),
                              gridspec_kw=dict(bottom=0.18, top=0.92,
                                              hspace=0.28, wspace=0.30,
                                              left=0.06, right=0.99))

    correlation_rows = []

    arch_handles_by_dataset: dict[str, list] = {}
    facet_handles_by_dataset: dict[str, list] = {}

    for col, (label, fname, facet_col) in enumerate(DATASETS):
        df = pd.read_csv(REPO / "results" / fname)
        slc = deepest_last_epoch_slice(df)
        slc = add_gaps(slc, facet_col)
        agg = aggregate(slc, facet_col)
        agg["dataset"] = label

        archs = sorted(agg["arch_str"].unique(),
                       key=lambda s: (parse_arch(s)[0], parse_arch(s)[1]))
        palette = make_palette(archs)
        facet_vals = sorted(agg[facet_col].unique())
        markers = ["o", "s", "D", "^", "v", "P", "X"]
        facet_marker_map = dict(zip(facet_vals, markers))

        for row, (ymetric, ylabel) in enumerate([
            ("gen_gap_norm", r"$\mathrm{train\_acc} - (1-n)\cdot\mathrm{test\_acc}$"),
            ("gen_gap_abs",  r"$|\mathrm{train\_acc} - \mathrm{test\_acc}|$"),
        ]):
            ax = axes[row, col]
            plot_scatter(ax, agg, facet_col, ymetric, ylabel, palette,
                         title=(f"{label}" if row == 0 else ""),
                         facet_marker_map=facet_marker_map,
                         show_xlabel=(row == 1))

            x = agg["rho_func"].to_numpy()
            y = agg[ymetric].to_numpy()
            full = bootstrap_corr(x, y)

            mask = agg["rho"] <= RHO_TRUST
            trust = bootstrap_corr(agg.loc[mask, "rho_func"].to_numpy(),
                                   agg.loc[mask, ymetric].to_numpy())

            txt = (f"all (n={full['n_cells']}): r={full['pearson']:+.2f} "
                   f"[{full['pearson_lo']:+.2f},{full['pearson_hi']:+.2f}]\n"
                   f"trust ρ≤{RHO_TRUST} (n={trust['n_cells']}): r={trust['pearson']:+.2f} "
                   f"[{trust['pearson_lo']:+.2f},{trust['pearson_hi']:+.2f}]")
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=8,
                    va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="0.7", alpha=0.85))

            correlation_rows.append({
                "dataset": label, "ymetric": ymetric, "subset": "all", **full,
            })
            correlation_rows.append({
                "dataset": label, "ymetric": ymetric, "subset": f"rho_le_{RHO_TRUST}",
                **trust,
            })

        arch_handles_by_dataset[label] = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=palette[a], markeredgecolor="black",
                       markersize=8, label=a)
            for a in archs
        ]
        facet_handles_by_dataset[label] = [
            plt.Line2D([0], [0], marker=facet_marker_map[v], linestyle="",
                       markerfacecolor="lightgray", markeredgecolor="black",
                       markersize=8, label=f"{facet_col}={v}")
            for v in facet_vals
        ]

    # Bottom legends, one per column.
    for col, (label, _, facet_col) in enumerate(DATASETS):
        bbox = axes[1, col].get_position()
        legend_y = 0.005
        legend_x_left = bbox.x0
        legend_x_right = bbox.x1
        # arch legend on the left half, facet on the right
        ah = arch_handles_by_dataset[label]
        fh = facet_handles_by_dataset[label]
        ncol_arch = 3 if len(ah) > 6 else (2 if len(ah) > 3 else 1)
        fig.legend(handles=ah, title=f"{label} arch", fontsize=7,
                   title_fontsize=8, ncol=ncol_arch,
                   loc="lower left",
                   bbox_to_anchor=(legend_x_left, legend_y),
                   frameon=True)
        fig.legend(handles=fh, title=f"{label} {facet_col}", fontsize=7,
                   title_fontsize=8,
                   loc="lower right",
                   bbox_to_anchor=(legend_x_right, legend_y),
                   frameon=True)

    fig.suptitle("Experiment 2 — ρ_func vs generalization gap "
                 "(deepest layer, ε = 10, last epoch; mean over 5 seeds per cell)",
                 fontsize=12, y=0.985)

    outdir = REPO / "figures"
    outdir.mkdir(exist_ok=True, parents=True)
    figpath = outdir / "rho_func_vs_gen_gap.png"
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figpath}")

    corr_df = pd.DataFrame(correlation_rows)
    corr_path = REPO / "results" / "rho_func_vs_gen_gap_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"wrote {corr_path}")

    print("\nCorrelations (cross-cell):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(corr_df.round(3).to_string(index=False))

    # ---- (b) within-noise-level robustness check ----
    print("\n\nWithin-facet correlations (controls for noise/target_dim being a confound):")
    within_rows = []
    for label, fname, facet_col in DATASETS:
        df = pd.read_csv(REPO / "results" / fname)
        slc = deepest_last_epoch_slice(df)
        slc = add_gaps(slc, facet_col)
        agg = aggregate(slc, facet_col)
        for facet_val, sub in agg.groupby(facet_col):
            for ymetric in ("gen_gap_norm", "gen_gap_abs"):
                stats = bootstrap_corr(sub["rho_func"].to_numpy(),
                                       sub[ymetric].to_numpy())
                within_rows.append({
                    "dataset": label, "facet_col": facet_col,
                    "facet_val": facet_val, "ymetric": ymetric, **stats,
                })
    within_df = pd.DataFrame(within_rows)
    within_path = REPO / "results" / "rho_func_vs_gen_gap_within.csv"
    within_df.to_csv(within_path, index=False)
    print(f"wrote {within_path}\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(within_df.round(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
