"""Per-epoch, per-layer routing-information trajectories at noise=0.

To keep the layer index comparable across architectures, this restricts
to architectures with 5 hidden layers (layer index 1..5 then has the
same depth meaning everywhere). Baseline MI estimators are shown as
per-epoch trajectories on all layers where data is available.

Inputs:
    results/mi_baselines.csv
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv

Outputs:
    figures/routing_per_epoch_per_layer_noise0.png
    results/routing_per_epoch_per_layer_summary.csv

Usage:
    uv run python scripts/plot_routing_per_epoch_per_layer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

LAYERS = [1, 2, 3, 4, 5]
DEEPEST_LAYER = 5
EPS_FUNC = 10.0
NOISE = 0.0

# Restrict to 5-hidden-layer arches so layer index 1..5 has consistent
# depth semantics across the pool (each dataset contributes 4 arches x
# 5 seeds = 20 cells per layer per epoch).
ARCHS_FIVE = {
    "composite": [
        "[5, 5, 5, 5, 5]",
        "[7, 7, 7, 7, 7]",
        "[9, 9, 9, 9, 9]",
        "[25, 25, 25, 25, 25]",
    ],
    "wbc": [
        "[5, 5, 5, 5, 5]",
        "[7, 7, 7, 7, 7]",
        "[9, 9, 9, 9, 9]",
        "[10, 10, 10, 10, 10]",
        "[25, 25, 25, 25, 25]",
    ],
}

DATASET_ORDER = ["composite", "wbc"]
DATASET_TITLE = {"composite": "Composite", "wbc": "WBC"}

OURS = [
    ("plug_in_bits", r"$\hat I_{\mathrm{raw}}$ (plug-in)", "tab:red", "-", "o"),
    ("miller_madow_bits", r"$\tilde I_{\mathrm{raw}}$ (M-M)", "tab:orange", "-", "s"),
    ("plug_in_func_bits", r"$\hat I_{\mathrm{func}}$ (plug-in)", "tab:blue", ":", "o"),
    ("miller_madow_func_bits", r"$\tilde I_{\mathrm{func}}$ (M-M)", "tab:green", ":", "s"),
]

BASELINES = [
    ("bits_binning_8", r"binning $K{=}8$", "tab:gray"),
    ("bits_kmeans_KKY", r"kmeans $K{=}|Y|$", "tab:olive"),
    ("bits_ksg_k3", r"KSG $k{=}3$", "tab:cyan"),
    ("bits_infonce_mean", "InfoNCE", "tab:purple"),
    ("bits_mine_mean", "MINE-f", "tab:brown"),
]


def load_ours(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{dataset}_label_noise_new_estimator.csv")
    return df[
        (df["noise_level"] == NOISE)
        & (np.isclose(df["epsilon"], EPS_FUNC))
        & (df["arch_str"].isin(ARCHS_FIVE[dataset]))
        & (df["layer"].isin(LAYERS))
    ].copy()


def load_baselines(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "mi_baselines.csv")
    return df[
        (df["dataset"] == dataset)
        & (df["noise_level"] == NOISE)
        & (df["arch_str"].isin(ARCHS_FIVE[dataset]))
        & (df["layer"].isin(LAYERS))
    ].copy()


def aggregate_ours(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label, *_ in OURS:
        grp = (
            df.groupby(["layer", "epoch"])[col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grp["estimator"] = label
        grp["estimator_col"] = col
        rows.append(grp)
    return pd.concat(rows, ignore_index=True).rename(
        columns={"mean": "mean_bits", "std": "std_bits", "count": "n_cells"}
    )


def aggregate_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate baseline estimators per layer, per epoch across seeds."""
    rows = []
    for col, label, *_ in BASELINES:
        grp = (
            df.groupby(["layer", "epoch"])[col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grp["estimator"] = label
        grp["estimator_col"] = col
        rows.append(grp)
    return pd.concat(rows, ignore_index=True).rename(
        columns={"mean": "mean_bits", "std": "std_bits", "count": "n_cells"}
    ) if rows else pd.DataFrame()


def plot(
    summaries: dict[str, pd.DataFrame],
    baseline_summaries: dict[str, pd.DataFrame],
    H_Y: dict[str, float],
    out_path: Path,
) -> None:
    datasets = [d for d in DATASET_ORDER if d in summaries]
    n_rows, n_cols = len(datasets), len(LAYERS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), sharex=True, sharey="row"
    )
    if n_rows == 1:
        axes = np.array([axes])

    for r, ds in enumerate(datasets):
        sub = summaries[ds]
        baselines_ds = baseline_summaries.get(ds)
        epochs_max = int(sub["epoch"].max())
        
        for c, layer in enumerate(LAYERS):
            ax = axes[r, c]
            cell = sub[sub["layer"] == layer]

            # Plot our estimators
            for col, label, colour, ls, marker in OURS:
                row = cell[cell["estimator_col"] == col].sort_values("epoch")
                if row.empty:
                    continue
                xs = row["epoch"].to_numpy()
                ys = row["mean_bits"].to_numpy()
                es = row["std_bits"].to_numpy()
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=colour,
                    linestyle=ls,
                    linewidth=1.5,
                    markersize=2.5,
                    label=label if (r == 0 and c == 0) else None,
                )
                ax.fill_between(
                    xs, ys - es, ys + es, color=colour, alpha=0.12, linewidth=0
                )

            # Plot baseline trajectories on all layers
            if baselines_ds is not None and not baselines_ds.empty:
                for col, blabel, bcolour in BASELINES:
                    row = baselines_ds[
                        (baselines_ds["layer"] == layer)
                        & (baselines_ds["estimator_col"] == col)
                    ].sort_values("epoch")
                    if not row.empty:
                        xs = row["epoch"].to_numpy()
                        ys = row["mean_bits"].to_numpy()
                        es = row["std_bits"].to_numpy()
                        ax.plot(
                            xs,
                            ys,
                            color=bcolour,
                            linestyle="--",
                            linewidth=1.0,
                            alpha=0.85,
                            label=blabel if (r == 0 and c == 0) else None,
                        )
                        ax.fill_between(
                            xs,
                            ys - es,
                            ys + es,
                            color=bcolour,
                            alpha=0.07,
                            linewidth=0,
                        )

            ax.axhline(H_Y[ds], color="k", linestyle="-.", linewidth=0.7, alpha=0.5)

            if r == 0:
                ax.set_title(f"layer {layer}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{DATASET_TITLE[ds]}\nbits")
            if r == n_rows - 1:
                ax.set_xlabel("epoch")
            ax.set_xlim(0, epochs_max)
            ax.grid(True, alpha=0.25)

        # H(Y) annotation on the rightmost panel of each row
        axes[r, -1].text(
            epochs_max,
            H_Y[ds],
            f"  $H(Y) = {H_Y[ds]:.2f}$",
            va="center",
            ha="left",
            fontsize=8,
            alpha=0.7,
        )

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        rf"Routing-information per epoch, per layer at $\eta = 0$,"
        rf" $\varepsilon = {EPS_FUNC:g}$ (5-hidden-layer arches)",
        y=1.10,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    summaries: dict[str, pd.DataFrame] = {}
    baseline_summaries: dict[str, pd.DataFrame] = {}
    H_Y: dict[str, float] = {}
    long_rows: list[pd.DataFrame] = []

    for ds in DATASET_ORDER:
        ours = load_ours(ds)
        if ours.empty:
            print(f"[warn] no ours rows for {ds}")
            continue
        summaries[ds] = aggregate_ours(ours)
        summaries[ds]["dataset"] = ds
        baselines = load_baselines(ds)
        baseline_summaries[ds] = aggregate_baselines(baselines)
        H_Y[ds] = float(ours["H_Y_bits"].mean())
        long_rows.append(summaries[ds].assign(dataset=ds))

    if not summaries:
        raise SystemExit("no datasets produced any rows")

    out_summary = RESULTS / "routing_per_epoch_per_layer_summary.csv"
    pd.concat(long_rows, ignore_index=True).to_csv(out_summary, index=False)
    print(f"wrote {out_summary}")

    plot(
        summaries,
        baseline_summaries,
        H_Y,
        FIGURES / "routing_per_epoch_per_layer_noise0.png",
    )


if __name__ == "__main__":
    main()
