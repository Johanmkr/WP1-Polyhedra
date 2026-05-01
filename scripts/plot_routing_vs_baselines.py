"""Join MI baselines with the routing-information estimator outputs and
plot all nine estimators side by side.

Inputs:
    results/mi_baselines.csv
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv

Outputs:
    results/routing_vs_baselines_summary.csv  (long-format, one row per
        estimator * dataset * noise; mean / std / n)
    figures/baseline_mi_with_plugin.png       (per-dataset panel of
        bits vs noise, one line per estimator, H(Y) dashed)

Restriction used for the join:
    deepest layer (layer 5) x last epoch (150) x epsilon = 10 for the
    functional-quotient variants.

Usage:
    uv run python scripts/plot_routing_vs_baselines.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

EPS_FUNC = 10.0

# Per-dataset "deepest hidden layer" and "last saved epoch" — in
# `mi_baselines.csv` each (dataset, arch, seed) row sits at its arch's
# deepest hidden layer; the deepest-deepest is 5 for the MLP runs and 4
# for LeNet5 (n_conv + n_fc_hidden = 2 + 2).
DEEP_LAYER_BY_DATASET = {"composite": 5, "wbc": 5, "mnist_full_lenet": 4}
LAST_EPOCH_BY_DATASET = {"composite": 150, "wbc": 150, "mnist_full_lenet": 100}

# Datasets to render. Each is panelled if it has at least one row in
# `mi_baselines.csv`; the ours-side estimator CSV (per-dataset
# `*_label_noise_new_estimator.csv`) is optional — if missing, the panel
# shows only the baseline curves with NaNs for the ours columns.
DATASET_ORDER = ["composite", "wbc", "mnist_full_lenet"]
DATASET_TITLE = {
    "composite": "Composite (2D, 7 cls)",
    "wbc": "WBC (UCI, 2 cls)",
    "mnist_full_lenet": "MNIST + LeNet (10 cls)",
}

ESTIMATORS = [
    # (csv column, display label, group, colour-style hint)
    ("bits_binning_8", r"binning $K{=}8$", "baseline", ("o", "tab:gray")),
    ("bits_kmeans_KKY", r"kmeans $K{=}|Y|$", "baseline", ("s", "tab:olive")),
    ("bits_ksg_k3", r"KSG $k{=}3$", "baseline", ("D", "tab:cyan")),
    ("bits_infonce_mean", "InfoNCE", "baseline", ("v", "tab:purple")),
    ("bits_mine_mean", "MINE-f", "baseline", ("^", "tab:brown")),
    ("plug_in_bits", r"$\hat I_{\mathrm{raw}}$ (plug-in)", "ours", ("o", "tab:red")),
    ("miller_madow_bits", r"$\tilde I_{\mathrm{raw}}$ (M-M)", "ours", ("s", "tab:orange")),
    ("plug_in_func_bits", r"$\hat I_{\mathrm{func}}$ (plug-in)", "ours-func", ("o", "tab:blue")),
    ("miller_madow_func_bits", r"$\tilde I_{\mathrm{func}}$ (M-M)", "ours-func", ("s", "tab:green")),
]

JOIN_KEYS = ["dataset", "noise_level", "arch_str", "seed", "epoch", "layer"]


def load_estimator_csv(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    deep = DEEP_LAYER_BY_DATASET.get(dataset, 5)
    last = LAST_EPOCH_BY_DATASET.get(dataset, 150)
    return df[
        (df["epoch"] == last)
        & (df["layer"] == deep)
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ].copy()


def build_merged() -> pd.DataFrame:
    mi_all = pd.read_csv(RESULTS / "mi_baselines.csv")
    mi_blocks = []
    for ds in DATASET_ORDER:
        deep = DEEP_LAYER_BY_DATASET.get(ds, 5)
        block = mi_all[(mi_all["dataset"] == ds) & (mi_all["layer"] == deep)].copy()
        if not block.empty:
            mi_blocks.append(block)
    if not mi_blocks:
        raise SystemExit("no rows matched DEEP_LAYER_BY_DATASET in mi_baselines.csv")
    mi = pd.concat(mi_blocks, ignore_index=True)

    pieces = []
    for ds in DATASET_ORDER:
        path = RESULTS / f"{ds}_label_noise_new_estimator.csv"
        if not path.exists():
            print(f"[warn] missing {path.name}; baselines for {ds} will plot without ours-side overlays")
            continue
        pieces.append(load_estimator_csv(path, ds))
    est = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=JOIN_KEYS)
    )

    merged = mi.merge(est, on=JOIN_KEYS, how="left", suffixes=("", "_est"))
    if merged.empty:
        raise SystemExit("merge produced no rows; check mi_baselines.csv")
    # When ours-side is missing, H_Y comes from the baselines side: derive it
    # from log2(num_classes) since `mi_baselines.csv` doesn't carry H_Y.
    if "H_Y_bits" not in merged.columns:
        merged["H_Y_bits"] = np.log2(merged["num_classes"].astype(float))
    else:
        fill = np.log2(merged["num_classes"].astype(float))
        merged["H_Y_bits"] = merged["H_Y_bits"].fillna(fill)
    return merged


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, eta), grp in merged.groupby(["dataset", "noise_level"]):
        for col, label, group, _ in ESTIMATORS:
            if col not in grp.columns:
                continue
            vals = grp[col].astype(float).to_numpy()
            n = int(np.isfinite(vals).sum())
            if n == 0:
                continue
            rows.append(
                {
                    "dataset": ds,
                    "noise_level": float(eta),
                    "estimator_col": col,
                    "estimator_label": label,
                    "group": group,
                    "mean_bits": float(np.nanmean(vals)),
                    "std_bits": float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0,
                    "n_cells": n,
                    "H_Y_bits": float(grp["H_Y_bits"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, merged: pd.DataFrame, out_path: Path) -> None:
    datasets = [d for d in DATASET_ORDER if d in summary["dataset"].unique()]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.4 * len(datasets), 4.4), sharey=False
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = summary[summary["dataset"] == ds]
        eta_vals = sorted(sub["noise_level"].unique())

        for col, label, group, (marker, colour) in ESTIMATORS:
            row = sub[sub["estimator_col"] == col].sort_values("noise_level")
            if row.empty:
                continue
            xs = row["noise_level"].to_numpy()
            ys = row["mean_bits"].to_numpy()
            es = row["std_bits"].to_numpy()
            ls = (
                "-"
                if group == "ours"
                else (":" if group == "ours-func" else "--")
            )
            lw = 2.0 if group.startswith("ours") else 1.2
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker=marker,
                color=colour,
                linestyle=ls,
                linewidth=lw,
                markersize=6,
                capsize=3,
                label=label,
            )

        h_y = float(merged[merged["dataset"] == ds]["H_Y_bits"].mean())
        ax.axhline(h_y, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            eta_vals[-1],
            h_y,
            f"  $H(Y) = {h_y:.2f}$",
            va="center",
            ha="left",
            fontsize=8,
            alpha=0.7,
        )

        ax.set_title(DATASET_TITLE.get(ds, ds))
        ax.set_xlabel(r"label noise $\eta$")
        ax.set_ylabel("bits")
        ax.set_xticks(eta_vals)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        rf"Routing-information estimators vs MI baselines"
        rf" (deepest hidden layer * last epoch, $\varepsilon={EPS_FUNC:g}$)",
        y=1.08,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    merged = build_merged()
    print(f"merged {len(merged)} cells across {merged['dataset'].nunique()} datasets")
    summary = summarize(merged)
    summary_path = RESULTS / "routing_vs_baselines_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    plot(summary, merged, FIGURES / "baseline_mi_with_plugin.png")


if __name__ == "__main__":
    main()
