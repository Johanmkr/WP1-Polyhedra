"""Panel A — calibration scatter: routing estimator vs each baseline.

For every (dataset, architecture, seed) at the last-epoch / deepest-layer
operating point we have one routing-estimator value and one value per baseline.
This script scatters them — diagonal y=x and Pearson r per panel — to show
that the routing estimator agrees with established methods across all
conditions, not at one cherry-picked point.

Inputs:
    results/mi_baselines.csv

Outputs:
    figures/calibration_scatter.png
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

LAST_EPOCH_BY_DATASET = {"composite": 150, "wbc": 150}
DATASETS = ["composite", "wbc"]

DATASET_COLOUR = {
    "composite": "tab:blue",
    "wbc": "tab:orange",
    "mnist_fc": "tab:green",
}
DATASET_LABEL = {
    "composite": "Composite (7 cls)",
    "wbc": "WBC (2 cls)",
    "mnist_fc": "MNIST FC (10 cls, PCA-10)",
}

# Restrict to the option-C architectures we ran in the targeted sweep.
ARCHS_C = [
    "[5, 5, 5]",
    "[5, 5, 5, 5, 5]",
    "[9, 9, 9]",
    "[9, 9, 9, 9, 9]",
    "[25, 25, 25]",
    "[25, 25, 25, 25, 25]",
]

# MNIST FC: only architectures with positive M-M routing bits at target_dim=10.
ARCHS_MNIST_FC = ["[3, 3, 3]", "[5, 5, 5]", "[7, 7, 7]"]
MNIST_FC_TARGET_DIM = 10
MNIST_FC_EPOCH = 150
MNIST_FC_LAYER = 3
MNIST_FC_EPS = 10.0

BASELINES = [
    ("bits_binning_8", r"binning $K{=}8$"),
    ("bits_kmeans_KKY", r"k-means $K{=}|Y|$"),
    ("bits_ksg_k3", r"KSG $k{=}3$"),
]

OURS_COL = "bits_ours_raw"          # M-M corrected routing estimator
OURS_LABEL = r"$\tilde I_{\mathrm{raw}}$ (routing, M-M)"


def _arch_depth(arch_str: str) -> int:
    """Number of hidden layers from a string like '[5, 5, 5, 5, 5]'."""
    return arch_str.count(",") + 1


def _load_composite_wbc() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "mi_baselines.csv")
    df = df[df["dataset"].isin(DATASETS) & df["arch_str"].isin(ARCHS_C)].copy()
    if df.empty:
        raise SystemExit(
            "No calibration rows found — has the targeted baseline sweep finished?"
        )
    df["depth"] = df["arch_str"].map(_arch_depth)
    keep = []
    for ds in DATASETS:
        sub = df[
            (df["dataset"] == ds)
            & (df["epoch"] == LAST_EPOCH_BY_DATASET[ds])
            & (df["layer"] == df["depth"])
        ]
        keep.append(sub)
    out = pd.concat(keep, ignore_index=True)
    return out[out[OURS_COL].notna()].copy()


def _load_mnist_fc() -> pd.DataFrame:
    """Merge MNIST FC baselines with routing estimator values."""
    bl_path = RESULTS / "mnist_fc_baselines.csv"
    est_path = RESULTS / "mnist_capacity_new_estimator.csv"
    if not bl_path.exists() or not est_path.exists():
        print("[warn] MNIST FC baselines or estimator CSV missing — skipping")
        return pd.DataFrame()

    bl = pd.read_csv(bl_path)
    est = pd.read_csv(est_path)
    est = est[
        (est["epoch"] == MNIST_FC_EPOCH)
        & (est["layer"] == MNIST_FC_LAYER)
        & (np.isclose(est["epsilon"], MNIST_FC_EPS))
        & (est["target_dim"] == MNIST_FC_TARGET_DIM)
        & (est["arch_str"].isin(ARCHS_MNIST_FC))
    ][["arch_str", "seed", "miller_madow_bits"]].copy()
    est = est.rename(columns={"miller_madow_bits": OURS_COL})

    merged = bl.merge(est, on=["arch_str", "seed"], how="inner")
    merged["dataset"] = "mnist_fc"
    merged["noise_level"] = 0.0
    return merged


def load_calibration_rows() -> pd.DataFrame:
    parts = [_load_composite_wbc(), _load_mnist_fc()]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(BASELINES), figsize=(4.0 * len(BASELINES), 4.0))
    for ax, (col, label) in zip(axes, BASELINES):
        sub = df[df[col].notna()].copy()
        if sub.empty:
            ax.set_title(f"{label} (no data)")
            continue
        x = sub[OURS_COL].to_numpy()
        y = sub[col].to_numpy()
        for ds, sub_ds in sub.groupby("dataset"):
            ax.scatter(
                sub_ds[OURS_COL],
                sub_ds[col],
                s=22,
                alpha=0.7,
                color=DATASET_COLOUR.get(ds, "tab:gray"),
                label=DATASET_LABEL.get(ds, ds),
                edgecolors="white",
                linewidths=0.4,
            )
        # Diagonal y=x and Pearson r over the pooled sample.
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            "k--",
            lw=0.8,
            alpha=0.5,
            label="y = x",
        )
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(
            0.04,
            0.96,
            rf"$r = {r:.3f}$  ($n={len(x)}$)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(OURS_LABEL + " [bits]")
        ax.set_ylabel(label + " [bits]")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Calibration: routing estimator vs baselines (last epoch, deepest layer)",
        y=1.08,
        fontsize=11,
    )
    # fig.tight_layout()
    # fig.savefig(out_path, dpi=200, bbox_inches="tight")
    # print(f"wrote {out_path}")
    savefig(fig, neurips_figpath / "calibration_scatter")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = load_calibration_rows()
    plot(df, FIGURES / "calibration_scatter.png")


if __name__ == "__main__":
    main()
