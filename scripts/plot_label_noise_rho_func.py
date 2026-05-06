"""Experiment B — Compression ratio ρ_func vs label noise.

ρ_func = |Ω_func| / |Ω| at last layer, last epoch as a function of noise η.
Expected: ρ_func increases toward 1 as η grows (less structured routing).

Inputs:
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv

Outputs:
    figures/label_noise_rho_func.png
    figures/label_noise_rho_func.pdf
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
LAST_LAYER = 5
EPS_FUNC = 1.0

ARCHS_FIVE = ["[5, 5, 5, 5, 5]", "[9, 9, 9, 9, 9]", "[25, 25, 25, 25, 25]"]

DATASETS = {
    "composite": ("Composite", "tab:blue"),
    "wbc": ("WBC", "tab:orange"),
}


def load_dataset(name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{name}_label_noise_new_estimator.csv")
    return df[
        (df["epoch"] == LAST_EPOCH)
        & (df["layer"] == LAST_LAYER)
        & (df["arch_str"].isin(ARCHS_FIVE))
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ].copy()


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))

    for name, (label, colour) in DATASETS.items():
        df = load_dataset(name)
        if df.empty:
            print(f"[warn] no data for {name}")
            continue
        agg = df.groupby("noise_level")["rho_func"].agg(["mean", "std"]).reset_index()
        xs = agg["noise_level"].to_numpy()
        ys = agg["mean"].to_numpy()
        es = agg["std"].fillna(0).to_numpy()
        ax.plot(xs, ys, color=colour, lw=2, marker="o", markersize=6, label=label)
        ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.15, lw=0)

    ax.axhline(1.0, color="k", linestyle="--", lw=0.8, alpha=0.4, label="ρ=1 (no compression)")
    ax.set_xlabel("Label noise rate η")
    ax.set_ylabel(r"$\rho_{\mathrm{func}} = |\Omega_{\mathrm{func}}| / |\Omega|$")
    ax.set_xlim(-0.02, 0.45)
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_title(rf"Compression ratio vs label noise (last layer, $\varepsilon={EPS_FUNC:g}$)")
    fig.tight_layout()
    savefig(fig, neurips_figpath / "label_noise_rho_func")


if __name__ == "__main__":
    main()
