"""Experiment E — ρ_func layerwise profile under label noise.

Same 3-panel layout as the main rho_func_layerwise figure, but with one line
per noise rate η instead of one line per width.  The V-shape at η=0 should
match the existing figure; it should flatten as η increases.

Inputs:
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv

Outputs:
    figures/label_noise_rho_func_layerwise.png
    figures/label_noise_rho_func_layerwise.pdf
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
RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"

LAST_EPOCH = 150
EPS_FUNC = 1.0

ARCHS_FIVE = ["[5, 5, 5, 5, 5]", "[9, 9, 9, 9, 9]", "[25, 25, 25, 25, 25]"]
LAYERS_FIVE = [1, 2, 3, 4, 5]

# Noise levels to show; one line per η
NOISE_LEVELS = [0.0, 0.2, 0.4]
NOISE_COLOURS = {0.0: "tab:blue", 0.2: "tab:orange", 0.4: "tab:red"}

DATASETS = {
    "composite": "Composite (7 cls)",
    "wbc": "WBC (2 cls)",
}


def load_dataset(name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{name}_label_noise_new_estimator.csv")
    return df[
        (df["epoch"] == LAST_EPOCH)
        & (df["arch_str"].isin(ARCHS_FIVE))
        & (df["layer"].isin(LAYERS_FIVE))
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ].copy()


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, title) in zip(axes, DATASETS.items()):
        df = load_dataset(name)
        if df.empty:
            print(f"[warn] no data for {name}")
            ax.set_title(title + " (no data)")
            continue

        for eta in NOISE_LEVELS:
            sub = df[np.isclose(df["noise_level"], eta)]
            if sub.empty:
                continue
            agg = sub.groupby("layer")["rho_func"].agg(["mean", "std"]).reset_index()
            xs = agg["layer"].to_numpy()
            ys = agg["mean"].to_numpy()
            es = agg["std"].fillna(0).to_numpy()
            colour = NOISE_COLOURS[eta]
            ax.plot(xs, ys, color=colour, lw=2, marker="o", markersize=5,
                    label=f"η={eta:.1f}")
            ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.12, lw=0)

        ax.axhline(1.0, color="k", linestyle="--", lw=0.7, alpha=0.4)
        ax.set_xticks(LAYERS_FIVE)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\rho_{\mathrm{func}}$")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle(
        r"$\rho_{\mathrm{func}}$ layerwise profile under label noise"
        rf" ($\varepsilon={EPS_FUNC:g}$, epoch {LAST_EPOCH})",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, neurips_figpath / "label_noise_rho_func_layerwise")


if __name__ == "__main__":
    main()
