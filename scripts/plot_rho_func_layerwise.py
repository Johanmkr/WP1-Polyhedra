"""ρ_func as a function of layer depth across all three datasets.

ρ_func = |Ω_{D,func}| / |Ω_D| ∈ (0,1] measures how much combinatorial
capacity is genuinely used vs. collapsed by functional equivalence.  Values
near 1 mean almost every activation pattern implements a distinct affine map;
values near 0 mean the network reuses a small set of affine maps across many
distinct routing paths.

One panel per dataset; within each panel one line per architecture width,
mean ± std across seeds.

Inputs:
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv
    results/mnist_capacity_new_estimator.csv

Outputs:
    figures/rho_func_layerwise.png  (also written to neurips_figpath)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
LAST_EPOCH = 150
EPS_FUNC = 10.0

ARCHS_FIVE = ["[5, 5, 5, 5, 5]", "[9, 9, 9, 9, 9]", "[25, 25, 25, 25, 25]"]
ARCHS_MNIST = ["[5, 5, 5]", "[7, 7, 7]"]
MNIST_TARGET_DIM = 10

DATASET_TITLE = {
    "composite": "Composite (7 cls, $N$=10k)",
    "wbc": "WBC (2 cls, $N$=569)",
    "mnist": "MNIST (10 cls, PCA-10)",
}

# Width labels for legend.
def _width(arch: str) -> int:
    return int(arch.strip()[1:-1].split(",")[0].strip())


def load_composite_wbc(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{dataset}_label_noise_new_estimator.csv")
    return df[
        (df["noise_level"] == NOISE)
        & (df["epoch"] == LAST_EPOCH)
        & (df["arch_str"].isin(ARCHS_FIVE))
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ][["arch_str", "layer", "rho_func", "seed"]].copy()


def load_mnist() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "mnist_capacity_new_estimator.csv")
    return df[
        (df["epoch"] == LAST_EPOCH)
        & (df["target_dim"] == MNIST_TARGET_DIM)
        & (df["arch_str"].isin(ARCHS_MNIST))
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ][["arch_str", "layer", "rho_func", "seed"]].copy()


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    archs: list[str],
) -> None:
    widths = sorted({_width(a) for a in archs})
    colours = cm.viridis(np.linspace(0.15, 0.85, len(widths)))
    colour_map = {w: c for w, c in zip(widths, colours)}

    for arch in archs:
        w = _width(arch)
        sub = df[df["arch_str"] == arch].groupby("layer")["rho_func"].agg(
            ["mean", "std"]
        ).reset_index()
        if sub.empty:
            continue
        xs = sub["layer"].to_numpy()
        ys = sub["mean"].to_numpy()
        es = sub["std"].to_numpy()
        colour = colour_map[w]
        ax.plot(xs, ys, marker="o", color=colour, lw=1.8, markersize=5,
                label=f"width {w}")
        ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.15, lw=0)

    ax.axhline(1.0, color="k", linestyle="--", lw=0.7, alpha=0.4)
    layers = sorted(df["layer"].unique())
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\rho_{\mathrm{func}}$")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    data = {
        "composite": (load_composite_wbc("composite"), ARCHS_FIVE),
        "wbc": (load_composite_wbc("wbc"), ARCHS_FIVE),
        "mnist": (load_mnist(), ARCHS_MNIST),
    }

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), sharey=False)

    for ax, (ds, (df, archs)) in zip(axes, data.items()):
        if df.empty:
            print(f"[warn] no data for {ds}")
            ax.set_visible(False)
            continue
        _plot_panel(ax, df, DATASET_TITLE[ds], archs)

    fig.suptitle(
        rf"Functional-quotient compression $\rho_{{\mathrm{{func}}}}$ by layer"
        rf" (epoch {LAST_EPOCH}, $\varepsilon={EPS_FUNC:g}$)",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    savefig(fig, neurips_figpath / "rho_func_layerwise")
    print("wrote rho_func_layerwise")


if __name__ == "__main__":
    main()
