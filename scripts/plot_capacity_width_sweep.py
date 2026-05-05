"""Experiment C — Architecture width vs raw MM saturation.

Left y-axis: tilde_I_raw at last layer, last epoch (mean ± 1σ across seeds).
Right y-axis: rho_func at same layer/epoch.
Vertical dashed line marks the width at which |Omega| first exceeds N/10
(fine-resolution warning: rho > 0.1).

Inputs:
    results/mi_baselines.csv

Outputs:
    figures/capacity_width_sweep.png
    figures/capacity_width_sweep.pdf
"""

from __future__ import annotations

import re
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
RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"

LAST_EPOCH = 150
NOISE = 0.0

# Datasets to show and their labels
DATASETS = {
    "composite": ("Composite", "tab:blue"),
    "wbc": ("WBC", "tab:orange"),
}

H_Y = {"composite": 2.640, "wbc": 0.953}


def _parse_width(arch_str: str) -> int | None:
    m = re.findall(r"\d+", str(arch_str))
    return int(m[0]) if m else None


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "mi_baselines.csv")
    df = df[(df["epoch"] == LAST_EPOCH) & (df["noise_level"] == NOISE)].copy()
    df["width"] = df["arch_str"].apply(_parse_width)
    df = df.dropna(subset=["width", "bits_ours_raw", "rho_func"])
    df["width"] = df["width"].astype(int)
    # last layer = max layer per network_id
    last_layer = df.groupby(["network_id", "seed"])["layer"].transform("max")
    df = df[df["layer"] == last_layer].copy()
    return df


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    df_all = load_data()

    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (ds, (label, colour)) in zip(axes, DATASETS.items()):
        df = df_all[df_all["dataset"] == ds]
        if df.empty:
            print(f"[warn] no data for {ds}")
            ax.set_title(f"{label} (no data)")
            continue

        agg = df.groupby("width").agg(
            mi_mean=("bits_ours_raw", "mean"),
            mi_std=("bits_ours_raw", "std"),
            rho_mean=("rho_func", "mean"),
            rho_std=("rho_func", "std"),
            rho_raw_mean=("rho", "mean"),
        ).reset_index()
        xs = agg["width"].to_numpy()

        ax2 = ax.twinx()

        # Left: raw MM MI
        ax.plot(xs, agg["mi_mean"], color=colour, lw=2, marker="o", markersize=5,
                label=r"$\tilde{I}_{\mathrm{raw}}$")
        ax.fill_between(xs,
                        agg["mi_mean"] - agg["mi_std"].fillna(0),
                        agg["mi_mean"] + agg["mi_std"].fillna(0),
                        color=colour, alpha=0.15, lw=0)
        ax.axhline(H_Y[ds], color=colour, linestyle="-.", lw=0.8, alpha=0.5)
        ax.text(xs[-1] * 1.02, H_Y[ds], "H(Y)", fontsize=8, color=colour, alpha=0.7)

        # Right: rho_func
        ax2.plot(xs, agg["rho_mean"], color="tab:red", lw=1.5, marker="s", markersize=4,
                 linestyle="--", label=r"$\rho_{\mathrm{func}}$")
        ax2.fill_between(xs,
                         agg["rho_mean"] - agg["rho_std"].fillna(0),
                         agg["rho_mean"] + agg["rho_std"].fillna(0),
                         color="tab:red", alpha=0.12, lw=0)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel(r"$\rho_{\mathrm{func}}$", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Fine-resolution warning line (rho > 0.1 → |Omega| > N/10)
        fine_res = agg[agg["rho_raw_mean"] > 0.1]
        if not fine_res.empty:
            w_thresh = int(fine_res["width"].iloc[0])
            ax.axvline(w_thresh, color="k", linestyle=":", lw=1.0, alpha=0.6,
                       label=f"fine-res (w={w_thresh})")

        ax.set_xlabel("Network width (neurons/layer)")
        ax.set_ylabel("Bits")
        ax.set_title(label)
        ax.grid(alpha=0.25)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.suptitle("Architecture width vs routing MI and compression (last layer, η=0)", fontsize=11)
    fig.tight_layout()
    savefig(fig, neurips_figpath / "capacity_width_sweep")


if __name__ == "__main__":
    main()
