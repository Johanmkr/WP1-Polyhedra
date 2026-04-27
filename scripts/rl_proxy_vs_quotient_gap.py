"""Experiment 3 — RL_proxy ↔ Ĩ_raw − Ĩ_func.

Prop 4.2 of the paper says the routing-loss term I(Y;Π|T) is positive iff
the data-supported RTG has at least one Hamming-1-adjacent pair of regions
whose dominant classes disagree. ``rl_proxy`` is the empirical fraction of
such adjacencies; ``Ĩ_raw − Ĩ_func`` (Miller-Madow corrected) is the
finite-data upper-bound surrogate on the routing-loss term (Sec 4.5).

Hypothesis: positive correlation across (arch, facet, seed) cells.

For each dataset, slice to deepest layer, ε = 10, last epoch. One row per
(arch, facet, seed). Aggregate per (arch, facet) cell; scatter
``Ĩ_raw − Ĩ_func`` vs ``rl_proxy``, colored by noise level / target_dim.
Report Pearson correlation per dataset with bootstrap CI.

Outputs:
- figures/rl_proxy_vs_quotient_gap.png
- results/rl_proxy_vs_quotient_gap_correlations.csv
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
    deepest = df.groupby("arch_str")["layer"].transform("max")
    sub = df[(df["layer"] == deepest) & (np.isclose(df["epsilon"], EPS))].copy()
    last_ep = sub.groupby(["arch_str", "seed"])["epoch"].transform("max")
    return sub[sub["epoch"] == last_ep].copy()


def aggregate(df: pd.DataFrame, facet_col: str) -> pd.DataFrame:
    df = df.assign(quotient_gap=df["miller_madow_bits"] - df["miller_madow_func_bits"])
    g = df.groupby(["arch_str", facet_col])
    out = g.agg(
        rl_proxy=("rl_proxy", "mean"),
        quotient_gap=("quotient_gap", "mean"),
        i_raw=("miller_madow_bits", "mean"),
        i_func=("miller_madow_func_bits", "mean"),
        rho=("rho", "mean"),
        rho_func=("rho_func", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return out


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n: int = N_BOOT) -> dict:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"pearson": np.nan, "pearson_lo": np.nan, "pearson_hi": np.nan,
                "spearman": np.nan, "spearman_lo": np.nan, "spearman_hi": np.nan,
                "n_cells": int(len(x))}
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
        "n_cells": int(len(x)),
    }


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                              gridspec_kw=dict(left=0.06, right=0.99,
                                              top=0.88, bottom=0.16, wspace=0.28))

    correlation_rows = []

    for col, (label, fname, facet_col) in enumerate(DATASETS):
        path = REPO / "results" / fname
        if not path.exists():
            axes[col].text(0.5, 0.5, f"{path.name}\nnot yet available",
                            ha="center", va="center", transform=axes[col].transAxes)
            continue
        df = pd.read_csv(path)
        if "rl_proxy" not in df.columns:
            axes[col].text(0.5, 0.5, f"{label}: rl_proxy column missing\n(rerun with new estimator)",
                            ha="center", va="center", transform=axes[col].transAxes)
            continue

        slc = deepest_last_epoch_slice(df)
        agg = aggregate(slc, facet_col)
        agg["dataset"] = label

        facet_vals = sorted(agg[facet_col].unique())
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(facet_vals), 2)))
        color_map = dict(zip(facet_vals, cmap))

        ax = axes[col]
        for facet_val, sub in agg.groupby(facet_col):
            ax.scatter(sub["rl_proxy"], sub["quotient_gap"],
                       color=color_map[facet_val], s=70,
                       edgecolor="black", linewidth=0.5,
                       label=f"{facet_col}={facet_val}", alpha=0.85)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel(r"$\mathrm{RL\_proxy}$ (data-supported RTG, deepest, last epoch)")
        ax.set_ylabel(r"$\tilde I_{\mathrm{raw}} - \tilde I_{\mathrm{func}}$  [bits]")
        ax.axhline(0, color="black", lw=0.6, ls=":", alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        x = agg["rl_proxy"].to_numpy()
        y = agg["quotient_gap"].to_numpy()
        full = bootstrap_corr(x, y)

        mask = (agg["rho"] <= RHO_TRUST).to_numpy()
        trust = bootstrap_corr(x[mask], y[mask])

        txt = (f"all (n={full['n_cells']}): r={full['pearson']:+.2f} "
               f"[{full['pearson_lo']:+.2f},{full['pearson_hi']:+.2f}]\n"
               f"trust ρ≤{RHO_TRUST} (n={trust['n_cells']}): r={trust['pearson']:+.2f} "
               f"[{trust['pearson_lo']:+.2f},{trust['pearson_hi']:+.2f}]")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=8,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.85))

        correlation_rows.append({"dataset": label, "subset": "all", **full})
        correlation_rows.append({"dataset": label,
                                  "subset": f"rho_le_{RHO_TRUST}", **trust})

    fig.suptitle("Experiment 3 — RL_proxy vs Ĩ_raw − Ĩ_func "
                 "(deepest layer, ε = 10, last epoch; mean over seeds per cell)",
                 fontsize=12)

    outdir = REPO / "figures"
    outdir.mkdir(exist_ok=True, parents=True)
    figpath = outdir / "rl_proxy_vs_quotient_gap.png"
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figpath}")

    if correlation_rows:
        corr_df = pd.DataFrame(correlation_rows)
        corr_path = REPO / "results" / "rl_proxy_vs_quotient_gap_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"wrote {corr_path}\n")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(corr_df.round(3).to_string(index=False))

    # ---- Within-facet stratification ----
    print("\n\nWithin-facet correlations (controls for the facet axis being a confound):")
    within_rows = []
    for label, fname, facet_col in DATASETS:
        path = REPO / "results" / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "rl_proxy" not in df.columns:
            continue
        slc = deepest_last_epoch_slice(df)
        agg = aggregate(slc, facet_col)
        for facet_val, sub in agg.groupby(facet_col):
            stats = bootstrap_corr(sub["rl_proxy"].to_numpy(),
                                   sub["quotient_gap"].to_numpy())
            within_rows.append({
                "dataset": label, "facet_col": facet_col,
                "facet_val": facet_val, **stats,
            })
    if within_rows:
        within_df = pd.DataFrame(within_rows)
        within_path = REPO / "results" / "rl_proxy_vs_quotient_gap_within.csv"
        within_df.to_csv(within_path, index=False)
        print(f"wrote {within_path}\n")
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(within_df.round(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
