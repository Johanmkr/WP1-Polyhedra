"""Panel C — per-epoch trajectory of the routing estimator (per layer, per
dataset).

Inputs:
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv

Outputs:
    figures/routing_trajectory.png
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
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

NOISE = 0.0
EPS_FUNC = 10.0
LAYERS = [1, 2, 3, 4, 5]

# 5-layer architectures only, so layer index has consistent semantics.
ARCHS_FIVE = [
    "[5, 5, 5, 5, 5]",
    "[9, 9, 9, 9, 9]",
    "[25, 25, 25, 25, 25]",
]

DATASETS = ["composite", "wbc"]
DATASET_TITLE = {"composite": "Composite", "wbc": "WBC"}

OURS = [
    ("plug_in_bits", r"$\hat I_{\mathrm{raw}}$ (plug-in)", "tab:red", "-", "o"),
    ("miller_madow_bits", r"$\tilde I_{\mathrm{raw}}$ (M-M)", "tab:orange", "-", "s"),
    ("plug_in_func_bits", r"$\hat I_{\mathrm{func}}$ (plug-in)", "tab:blue", ":", "o"),
    ("miller_madow_func_bits", r"$\tilde I_{\mathrm{func}}$ (M-M)", "tab:green", ":", "s"),
]


def load_ours(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{dataset}_label_noise_new_estimator.csv")
    return df[
        (df["noise_level"] == NOISE)
        & (df["arch_str"].isin(ARCHS_FIVE))
        & (df["layer"].isin(LAYERS))
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ].copy()


def aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grp = df.groupby(["layer", "epoch"])[value_col].agg(["mean", "std", "count"]).reset_index()
    return grp.rename(columns={"mean": "mean_bits", "std": "std_bits", "count": "n"})


def plot(
    ours: dict[str, pd.DataFrame],
    H_Y: dict[str, float],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(DATASETS),
        len(LAYERS),
        figsize=(2.6 * len(LAYERS), 2.8 * len(DATASETS)),
        sharex=True,
        sharey="row",
    )
    if len(DATASETS) == 1:
        axes = np.array([axes])

    for r, ds in enumerate(DATASETS):
        ours_ds = ours[ds]
        epoch_max = int(ours_ds["epoch"].max())

        for c, layer in enumerate(LAYERS):
            ax = axes[r, c]

            for col, label, colour, ls, marker in OURS:
                sub = ours_ds[ours_ds["layer"] == layer]
                if sub.empty:
                    continue
                agg = aggregate(sub, col)
                xs = agg["epoch"].to_numpy()
                ys = agg["mean_bits"].to_numpy()
                es = agg["std_bits"].to_numpy()
                ax.plot(xs, ys, color=colour, linestyle=ls, lw=1.4,
                        marker=marker, markersize=2.5,
                        label=label if (r == 0 and c == 0) else None)
                ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.10, lw=0)

            if ds in H_Y:
                ax.axhline(H_Y[ds], color="k", linestyle="-.", lw=0.7, alpha=0.5)

            if r == 0:
                ax.set_title(f"layer {layer}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{DATASET_TITLE[ds]}\nbits")
            if r == len(DATASETS) - 1:
                ax.set_xlabel("epoch")
            ax.set_xlim(0, epoch_max)
            ax.grid(alpha=0.25)

        if ds in H_Y:
            axes[r, -1].text(
                epoch_max,
                H_Y[ds],
                f"  $H(Y)={H_Y[ds]:.2f}$",
                va="center",
                ha="left",
                fontsize=8,
                alpha=0.7,
            )

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(
        handles, labels,
        loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9,
    )
    fig.suptitle(
        rf"Routing-estimator trajectories ($\eta=0$, 5-layer archs, "
        rf"$\varepsilon={EPS_FUNC:g}$)",
        y=1.10, fontsize=11,
    )
    # fig.tight_layout()
    # fig.savefig(out_path, dpi=200, bbox_inches="tight")
    # print(f"wrote {out_path}")
    savefig(fig, neurips_figpath / "routing_trajectory")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    ours, H_Y = {}, {}
    for ds in DATASETS:
        d = load_ours(ds)
        if d.empty:
            print(f"[warn] no routing rows for {ds}")
            continue
        ours[ds] = d
        H_Y[ds] = float(d["H_Y_bits"].mean())
    if not ours:
        raise SystemExit("no datasets to plot")
    plot(ours, H_Y, FIGURES / "routing_trajectory.png")


if __name__ == "__main__":
    main()
