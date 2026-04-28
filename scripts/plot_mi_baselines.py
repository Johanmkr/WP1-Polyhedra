"""Phase A.1 — MI baselines comparison plots.

Reads `results/mi_baselines.csv` (per-HDF5 MI baselines: binning, k-means,
KSG, InfoNCE, MINE-f, plus our `bits_ours_raw` and `bits_ours_func` joined
in) and `results/gen_gap_predictors.csv` (for `gen_gap_acc`).

Produces two figures and a summary CSV:

- `figures/baseline_mi_comparison.png` — three-panel main figure
  (planning §A.1.6):
    A. scatter `bits_<baseline>` vs `bits_ours_raw`, identity line, faceted
       by dataset. Filled markers = trustworthy (ρ ≤ 0.3); faded crosses =
       untrusted.
    B. within-noise Pearson r between each baseline and `gen_gap_acc`, as
       a `(method × noise_level)` heatmap per dataset.
    C. wall-clock (log-x) vs |bits − bits_MINE| accuracy. Each baseline
       gets one (median wall, median |Δbits|) point per dataset, with IQR
       error bars.

- `figures/baseline_mi_vs_noise.png` — DPI/noise diagnostic. For each
  (dataset, baseline), mean ± seed-std of `bits` vs `noise_level`. Lets the
  reader read off monotonicity at a glance.

- `results/baseline_mi_summary.csv` — per-(dataset, baseline, noise)
  Pearson r vs gen_gap_acc plus per-(dataset, baseline) median wall and
  median |Δbits_MINE|.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
RHO_TRUST = 0.3

# (key in CSV → display label, wall-seconds column or None)
BASELINES = [
    ("bits_ours_raw",    "ours_raw",    None),
    ("bits_ours_func",   "ours_func",   None),
    ("bits_binning_8",   "binning_8",   "wall_seconds_binning"),
    ("bits_kmeans_KKY",  "kmeans_|Y|",  "wall_seconds_kmeans"),
    ("bits_ksg_k3",      "ksg_k3",      "wall_seconds_ksg"),
    ("bits_infonce_mean","infonce",     "wall_seconds_infonce"),
    ("bits_mine_mean",   "mine",        "wall_seconds_mine"),
]
BASELINE_COLORS = {
    "ours_raw":  "#d62728",
    "ours_func": "#9467bd",
    "binning_8": "#1f77b4",
    "kmeans_|Y|":"#2ca02c",
    "ksg_k3":    "#8c564b",
    "infonce":   "#e377c2",
    "mine":      "#ff7f0e",
}


def load() -> pd.DataFrame:
    mi = pd.read_csv(REPO / "results" / "mi_baselines.csv")
    gg = pd.read_csv(REPO / "results" / "gen_gap_predictors.csv")
    keys = ["network_id", "dataset", "noise_level", "arch_str", "seed"]
    df = mi.merge(gg[keys + ["gen_gap_acc", "final_test_accuracy"]],
                  on=keys, how="left")
    df["trust"] = df["rho"] <= RHO_TRUST
    return df


# ---------------------------------------------------------------- panel A
def panel_a(ax, df: pd.DataFrame, dataset: str) -> None:
    sub = df[df["dataset"] == dataset]
    x_all = sub["bits_ours_raw"].to_numpy()
    lo = float(np.nanmin(x_all))
    hi = float(np.nanmax(x_all))
    for col, label, _ in BASELINES:
        if label == "ours_raw":  # x-axis reference
            continue
        if col not in sub.columns:
            continue
        c = BASELINE_COLORS[label]
        trust = sub[sub["trust"]]
        untrust = sub[~sub["trust"]]
        ax.scatter(untrust["bits_ours_raw"], untrust[col],
                   c=c, marker="x", s=18, alpha=0.25, linewidth=0.7)
        ax.scatter(trust["bits_ours_raw"], trust[col],
                   c=c, marker="o", s=22, alpha=0.7,
                   edgecolor="black", linewidth=0.3, label=label)
        hi = max(hi, float(np.nanmax(sub[col])))
        lo = min(lo, float(np.nanmin(sub[col])))
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, ls="--", label="y = x")
    ax.set_xlabel(r"$\widetilde{I}_{\mathrm{ours,raw}}(Y;\Omega)$  [bits]")
    ax.set_ylabel(r"$\widehat{I}_{\mathrm{baseline}}(Y;T)$  [bits]")
    ax.set_title(f"A. {dataset}: baselines vs ours", fontsize=10)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------- panel B
def panel_b(ax, df: pd.DataFrame, dataset: str,
            noises: list[float]) -> pd.DataFrame:
    sub = df[df["dataset"] == dataset]
    methods = [lbl for _, lbl, _ in BASELINES]
    cols = [c for c, _, _ in BASELINES]
    M = np.full((len(methods), len(noises)), np.nan)
    rows: list[dict] = []
    for i, (col, label) in enumerate(zip(cols, methods)):
        if col not in sub.columns:
            continue
        for j, n in enumerate(noises):
            cell = sub[(sub["noise_level"] == n)
                       & sub[col].notna()
                       & sub["gen_gap_acc"].notna()]
            if len(cell) >= 4 and cell[col].std() > 0 \
                    and cell["gen_gap_acc"].std() > 0:
                r = pearsonr(cell[col], cell["gen_gap_acc"]).statistic
                M[i, j] = r
                rows.append({
                    "dataset": dataset, "method": label,
                    "noise_level": n, "pearson_r": r, "n_cells": len(cell),
                })
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f"n={n}" for n in noises])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("noise level")
    ax.set_title(f"B. {dataset}: r(MI, gen_gap_acc), within-noise",
                 fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color=("white" if abs(M[i, j]) > 0.55 else "black"))
    return pd.DataFrame(rows), im


# ---------------------------------------------------------------- panel C
def panel_c(ax, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    sub = df[df["dataset"] == dataset].copy()
    rows: list[dict] = []
    for col, label, wall_col in BASELINES:
        if wall_col is None or col not in sub.columns or wall_col not in sub.columns:
            continue
        delta = (sub[col] - sub["bits_mine_mean"]).abs()
        wall = sub[wall_col]
        mask = delta.notna() & wall.notna()
        if mask.sum() == 0:
            continue
        med_wall = float(np.median(wall[mask]))
        med_delta = float(np.median(delta[mask]))
        wall_lo, wall_hi = np.percentile(wall[mask], [25, 75])
        d_lo, d_hi = np.percentile(delta[mask], [25, 75])
        c = BASELINE_COLORS[label]
        ax.errorbar(med_wall, med_delta,
                    xerr=[[med_wall - wall_lo], [wall_hi - med_wall]],
                    yerr=[[med_delta - d_lo], [d_hi - med_delta]],
                    fmt="o", color=c, ecolor=c, elinewidth=0.8,
                    markersize=8, markeredgecolor="black",
                    markeredgewidth=0.4, capsize=2, label=label)
        rows.append({
            "dataset": dataset, "method": label,
            "wall_med": med_wall, "wall_iqr_lo": float(wall_lo),
            "wall_iqr_hi": float(wall_hi),
            "abs_delta_mine_med": med_delta,
            "abs_delta_mine_iqr_lo": float(d_lo),
            "abs_delta_mine_iqr_hi": float(d_hi),
        })
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock per HDF5 [s, log]")
    ax.set_ylabel(r"$|\widehat{I}_{\mathrm{method}} - \widehat{I}_{\mathrm{MINE}}|$  [bits]")
    ax.set_title(f"C. {dataset}: cost vs MINE-agreement", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.axhline(0, color="black", lw=0.6, ls=":", alpha=0.5)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- diagnostic
def plot_mi_vs_noise(df: pd.DataFrame, datasets: list[str],
                     noises: list[float], outpath: Path) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.4),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        for col, label, _ in BASELINES:
            if col not in sub.columns:
                continue
            mu = sub.groupby("noise_level")[col].mean()
            sd = sub.groupby("noise_level")[col].std()
            mu = mu.reindex(noises)
            sd = sd.reindex(noises)
            c = BASELINE_COLORS[label]
            lw = 2.0 if "ours" in label else 1.2
            ls = "-" if "ours" in label else "--"
            ax.plot(mu.index, mu.values, color=c, lw=lw, ls=ls,
                    marker="o", markersize=5, label=label)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.10)
        ax.set_xlabel("label noise")
        ax.set_ylabel(r"$\widehat{I}(Y;T)$  [bits]")
        ax.set_title(f"{dataset}: MI vs noise (mean ± std over arch×seed)",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


# ----------------------------------------------------------------- main
def main() -> int:
    df = load()
    datasets = sorted(df["dataset"].unique())
    noises = sorted(df["noise_level"].unique())

    n_ds = len(datasets)
    fig, axes = plt.subplots(
        2, n_ds, figsize=(5.8 * n_ds, 9.6),
        gridspec_kw=dict(hspace=0.40, wspace=0.30,
                         left=0.08, right=0.97, top=0.88, bottom=0.07),
    )
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    # A on top row, C on bottom row. B becomes a separate side figure to
    # avoid cramping; we still report Panel B numbers in the summary CSV.
    for j, dataset in enumerate(datasets):
        panel_a(axes[0, j], df, dataset)
    for j, dataset in enumerate(datasets):
        c_rows = panel_c(axes[1, j], df, dataset)
        if j == 0:
            c_rows_all = c_rows
        else:
            c_rows_all = pd.concat([c_rows_all, c_rows], ignore_index=True)

    fig.suptitle(
        "Phase A.1 — MI baselines vs ours (deepest layer, last epoch)\n"
        "Panel A: filled = trustworthy (ρ ≤ 0.3), faded × = untrusted.   "
        "Panel C: medians; error bars = IQR.",
        fontsize=10, y=0.985,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.935), ncol=len(labels),
               fontsize=8, frameon=False)

    figpath = REPO / "figures" / "baseline_mi_comparison.png"
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figpath}")

    # Panel B as its own figure (heatmap per dataset).
    fig_b, axes_b = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 4.6),
                                 gridspec_kw=dict(left=0.10, right=0.92,
                                                  top=0.88, bottom=0.18,
                                                  wspace=0.25))
    if n_ds == 1:
        axes_b = [axes_b]
    b_rows_all = pd.DataFrame()
    last_im = None
    for ax, dataset in zip(axes_b, datasets):
        b_rows, im = panel_b(ax, df, dataset, noises)
        b_rows_all = pd.concat([b_rows_all, b_rows], ignore_index=True)
        last_im = im
    cbar = fig_b.colorbar(last_im, ax=axes_b, fraction=0.04, pad=0.04)
    cbar.set_label("Pearson r")
    fig_b.suptitle(
        "Phase A.1 — Panel B: within-noise Pearson r between MI and "
        "gen_gap_acc",
        fontsize=11,
    )
    figpath_b = REPO / "figures" / "baseline_mi_panel_b.png"
    fig_b.savefig(figpath_b, dpi=140, bbox_inches="tight")
    plt.close(fig_b)
    print(f"wrote {figpath_b}")

    # Diagnostic: MI vs noise.
    plot_mi_vs_noise(df, datasets, noises,
                     REPO / "figures" / "baseline_mi_vs_noise.png")

    # Summary CSV.
    summary_path = REPO / "results" / "baseline_mi_summary.csv"
    b_rows_all["panel"] = "B_within_noise_pearson"
    c_rows_all["panel"] = "C_cost_vs_mine_agreement"
    summary = pd.concat([b_rows_all, c_rows_all], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print("\nPanel B (Pearson r vs gen_gap_acc, within noise):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(b_rows_all.round(3).to_string(index=False))
    print("\nPanel C (cost vs MINE-agreement, medians ± IQR):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(c_rows_all.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
