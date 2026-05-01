"""Per-epoch routing-information trajectories at noise=0, layer=deepest,
with MI baselines as horizontal reference bands (baselines are only
available at the last epoch in `results/mi_baselines.csv`).

Inputs:
    results/mi_baselines.csv
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv

Outputs:
    figures/routing_per_epoch_noise0.png
    results/routing_per_epoch_summary.csv

Usage:
    uv run python scripts/plot_routing_per_epoch.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

DEEP_LAYER = 5
EPS_FUNC = 10.0
NOISE = 0.0

DATASET_ORDER = ["composite", "wbc"]
DATASET_TITLE = {
    "composite": "Composite (2D, 7 cls)",
    "wbc": "WBC (UCI, 2 cls)",
}

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
        & (df["layer"] == DEEP_LAYER)
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ].copy()


def load_baselines(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "mi_baselines.csv")
    return df[
        (df["dataset"] == dataset)
        & (df["noise_level"] == NOISE)
        & (df["layer"] == DEEP_LAYER)
    ].copy()


def aggregate_ours(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label, *_ in OURS:
        grp = df.groupby("epoch")[col].agg(["mean", "std", "count"]).reset_index()
        grp["estimator"] = label
        grp["estimator_col"] = col
        rows.append(grp)
    return pd.concat(rows, ignore_index=True).rename(
        columns={"mean": "mean_bits", "std": "std_bits", "count": "n_cells"}
    )


def aggregate_baselines(df: pd.DataFrame) -> dict[str, tuple[float, float, int]]:
    out = {}
    for col, *_ in BASELINES:
        vals = df[col].astype(float).to_numpy()
        out[col] = (
            float(np.nanmean(vals)),
            float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            int(np.isfinite(vals).sum()),
        )
    return out


def plot(
    summaries: dict[str, pd.DataFrame],
    baseline_summaries: dict[str, dict[str, tuple[float, float, int]]],
    H_Y: dict[str, float],
    out_path: Path,
) -> None:
    datasets = [d for d in DATASET_ORDER if d in summaries]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6.0 * len(datasets), 4.6), sharey=False
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = summaries[ds]
        epochs_max = int(sub["epoch"].max())

        for col, label, colour, ls, marker in OURS:
            row = sub[sub["estimator_col"] == col].sort_values("epoch")
            xs = row["epoch"].to_numpy()
            ys = row["mean_bits"].to_numpy()
            es = row["std_bits"].to_numpy()
            ax.plot(
                xs,
                ys,
                marker=marker,
                color=colour,
                linestyle=ls,
                linewidth=1.8,
                markersize=4,
                label=label,
            )
            ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.12, linewidth=0)

        for col, blabel, bcolour in BASELINES:
            mu, sd, _ = baseline_summaries[ds][col]
            ax.axhline(mu, color=bcolour, linestyle="--", linewidth=1.0, alpha=0.85, label=blabel)
            ax.fill_between(
                [0, epochs_max],
                mu - sd,
                mu + sd,
                color=bcolour,
                alpha=0.07,
                linewidth=0,
            )

        ax.axhline(H_Y[ds], color="k", linestyle="-.", linewidth=0.9, alpha=0.6)
        ax.text(
            epochs_max,
            H_Y[ds],
            f"  $H(Y) = {H_Y[ds]:.2f}$",
            va="center",
            ha="left",
            fontsize=8,
            alpha=0.7,
        )

        ax.set_title(DATASET_TITLE.get(ds, ds))
        ax.set_xlabel("epoch")
        ax.set_ylabel("bits")
        ax.set_xlim(0, epochs_max)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        rf"Routing-information trajectories at $\eta = 0$, layer={DEEP_LAYER}"
        rf" ($\varepsilon = {EPS_FUNC:g}$)."
        " Baselines (dashed) shown at last epoch only.",
        y=1.13,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    summaries: dict[str, pd.DataFrame] = {}
    baseline_summaries: dict[str, dict[str, tuple[float, float, int]]] = {}
    H_Y: dict[str, float] = {}
    long_rows: list[pd.DataFrame] = []

    for ds in DATASET_ORDER:
        ours = load_ours(ds)
        if ours.empty:
            print(f"[warn] no ours rows for {ds} at noise={NOISE} layer={DEEP_LAYER}")
            continue
        summaries[ds] = aggregate_ours(ours)
        summaries[ds]["dataset"] = ds
        baseline_summaries[ds] = aggregate_baselines(load_baselines(ds))
        H_Y[ds] = float(ours["H_Y_bits"].mean())
        long_rows.append(summaries[ds].assign(dataset=ds))

    if not summaries:
        raise SystemExit("no datasets produced any rows")

    out_summary = RESULTS / "routing_per_epoch_summary.csv"
    pd.concat(long_rows, ignore_index=True).to_csv(out_summary, index=False)
    print(f"wrote {out_summary}")

    plot(summaries, baseline_summaries, H_Y, FIGURES / "routing_per_epoch_noise0.png")


if __name__ == "__main__":
    main()
