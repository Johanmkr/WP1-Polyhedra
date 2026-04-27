"""
Plotting utilities for the routing / quotient / RTG estimator output.

Operates on either a per-HDF5 CSV (``new_estimator_seed_<seed>.csv``) or the
aggregated CSV produced by ``run_label_noise_estimator.py --aggregate``.

CLI examples
------------
Single-experiment dashboard (the test run we just produced)::

    uv run python visualization/plot_new_estimator.py \\
        --csv 'outputs/composite_label_noise/n0.0_[25, 25, 25, 25, 25]/new_estimator_seed_101.csv' \\
        --epsilon 10.0 \\
        --outdir figures/label_noise_new_estimator/composite_n0.0_25x5_seed101

Aggregate label-noise comparison (after a full sweep)::

    uv run python visualization/plot_new_estimator.py \\
        --csv results/label_noise_new_estimator.csv \\
        --noise-compare --dataset composite --arch '[25, 25, 25]' \\
        --outdir figures/label_noise_new_estimator/noise_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
PALETTE = "viridis"


# ---------------------------------------------------------------------------
# Single-experiment dashboard
# ---------------------------------------------------------------------------
def plot_mi_trajectories(
    df: pd.DataFrame, epsilon: float, ax: plt.Axes
) -> None:
    """MI vs. epoch per layer at a fixed ε. Plug-in, MM, and quotient MM."""
    sub = df[np.isclose(df["epsilon"], epsilon)]
    if sub.empty:
        ax.set_title(f"(no rows at ε={epsilon})")
        return
    layers = sorted(sub["layer"].unique())
    cmap = sns.color_palette(PALETTE, n_colors=len(layers))
    for color, l in zip(cmap, layers):
        s = sub[sub["layer"] == l].sort_values("epoch")
        ax.plot(s["epoch"], s["plug_in_bits"], color=color, linestyle=":", alpha=0.5)
        ax.plot(s["epoch"], s["miller_madow_bits"], color=color, linestyle="--", alpha=0.7)
        ax.plot(
            s["epoch"], s["miller_madow_func_bits"], color=color, linestyle="-",
            label=f"L{l}",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("bits")
    ax.set_title(f"MI trajectories (ε={epsilon})\n: plug-in, --: MM, ─: MM-quotient")
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_rho(df: pd.DataFrame, ax: plt.Axes) -> None:
    """ρ = R/N over epochs per layer (independent of ε; ε=0 row is fine)."""
    sub = df[df["epsilon"] == df["epsilon"].min()]
    layers = sorted(sub["layer"].unique())
    cmap = sns.color_palette(PALETTE, n_colors=len(layers))
    for color, l in zip(cmap, layers):
        s = sub[sub["layer"] == l].sort_values("epoch")
        ax.plot(s["epoch"], s["rho"], color=color, label=f"L{l}", marker="o", ms=3)
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="ρ=0.3 limit")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ρ = R / N")
    ax.set_title("Trustworthiness check (ρ)")
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_quotient_collapse(
    df: pd.DataFrame, epoch: Optional[int], ax: plt.Axes
) -> None:
    """ρ_func vs ε per layer at one epoch (or final if not given)."""
    if epoch is None:
        epoch = int(df["epoch"].max())
    sub = df[df["epoch"] == epoch].copy()
    sub = sub[sub["epsilon"] > 0]  # log-x, drop ε=0
    if sub.empty:
        ax.set_title(f"(no rows at epoch={epoch})")
        return
    layers = sorted(sub["layer"].unique())
    cmap = sns.color_palette(PALETTE, n_colors=len(layers))
    for color, l in zip(cmap, layers):
        s = sub[sub["layer"] == l].sort_values("epsilon")
        ax.plot(s["epsilon"], s["rho_func"], color=color, label=f"L{l}", marker="o", ms=3)
    ax.set_xscale("log")
    ax.set_xlabel("ε")
    ax.set_ylabel("ρ_func = num_quotient / R")
    ax.set_title(f"Functional collapse curve (epoch={epoch})")
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_rtg(df: pd.DataFrame, ax: plt.Axes) -> None:
    """RTG largest-component fraction and isolated fraction over epochs per layer."""
    sub = df[df["epsilon"] == df["epsilon"].min()]
    layers = sorted(sub["layer"].unique())
    cmap = sns.color_palette(PALETTE, n_colors=len(layers))
    for color, l in zip(cmap, layers):
        s = sub[sub["layer"] == l].sort_values("epoch")
        ax.plot(s["epoch"], s["rtg_largest_component_frac"], color=color,
                linestyle="-", label=f"L{l} largest")
        ax.plot(s["epoch"], s["rtg_isolated_frac"], color=color,
                linestyle=":", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction")
    ax.set_title("RTG: ─ largest comp.   ⋯ isolated")
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_truncation(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Truncation probability over epochs per layer (ε-independent)."""
    sub = df[df["epsilon"] == df["epsilon"].min()]
    if sub["truncation_prob"].isna().all():
        ax.set_title("Truncation prob: not available (no holdout set)")
        return
    layers = sorted(sub["layer"].unique())
    cmap = sns.color_palette(PALETTE, n_colors=len(layers))
    for color, l in zip(cmap, layers):
        s = sub[sub["layer"] == l].sort_values("epoch")
        ax.plot(s["epoch"], s["truncation_prob"], color=color, label=f"L{l}",
                marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("P(unseen pattern)")
    ax.set_title("Truncation probability")
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_dpi_check(df: pd.DataFrame, epsilon: float, ax: plt.Axes) -> None:
    """Sanity: plug_in_func ≤ plug_in across all rows at the chosen ε."""
    sub = df[np.isclose(df["epsilon"], epsilon)]
    if sub.empty:
        ax.set_title("(no rows)")
        return
    ax.scatter(sub["plug_in_bits"], sub["plug_in_func_bits"],
               c=sub["layer"], cmap=PALETTE, s=12, alpha=0.7)
    lim = max(sub["plug_in_bits"].max(), sub["plug_in_func_bits"].max())
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="DPI: y=x")
    ax.set_xlabel("plug_in_bits (raw)")
    ax.set_ylabel("plug_in_func_bits (quotient)")
    ax.set_title(f"DPI check (ε={epsilon})")
    ax.legend(loc="best", fontsize=8)


def make_dashboard(
    df: pd.DataFrame, epsilon: float, epoch_for_collapse: Optional[int],
    title: str, outpath: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_mi_trajectories(df, epsilon, axes[0, 0])
    plot_rho(df, axes[0, 1])
    plot_quotient_collapse(df, epoch_for_collapse, axes[0, 2])
    plot_rtg(df, axes[1, 0])
    plot_truncation(df, axes[1, 1])
    plot_dpi_check(df, epsilon, axes[1, 2])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outpath}")


# ---------------------------------------------------------------------------
# Aggregate: noise-level comparison
# ---------------------------------------------------------------------------
def plot_noise_compare(
    df: pd.DataFrame, epsilon: float, layer: int,
    title: str, outpath: Path,
) -> None:
    """One panel per metric, x=epoch, hue=noise_level (mean ± std over seeds)."""
    sub = df[(np.isclose(df["epsilon"], epsilon)) & (df["layer"] == layer)].copy()
    if sub.empty:
        print(f"no rows for ε={epsilon}, layer={layer}")
        return

    metrics = [
        ("miller_madow_bits", "I(Y;Ω) MM-corrected"),
        ("miller_madow_func_bits", "I_func(Y;Ω) MM-corrected"),
        ("rho_func", "ρ_func = |quotient|/R"),
        ("rtg_largest_component_frac", "RTG largest component frac"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    noise_levels = sorted(sub["noise_level"].unique())
    cmap = sns.color_palette("rocket", n_colors=len(noise_levels))

    for ax, (col, label) in zip(axes.flat, metrics):
        for color, n in zip(cmap, noise_levels):
            s = sub[sub["noise_level"] == n]
            agg = s.groupby("epoch")[col].agg(["mean", "std"]).reset_index()
            ax.plot(agg["epoch"], agg["mean"], color=color, label=f"noise={n}", marker="o", ms=3)
            if agg["std"].notna().any():
                ax.fill_between(
                    agg["epoch"],
                    agg["mean"] - agg["std"],
                    agg["mean"] + agg["std"],
                    color=color, alpha=0.2,
                )
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="per-HDF5 or aggregated CSV")
    p.add_argument("--epsilon", type=float, default=10.0,
                   help="ε at which to slice MI / DPI plots (default 10.0)")
    p.add_argument("--collapse-epoch", type=int, default=None,
                   help="epoch for ρ_func vs ε collapse plot (default: max)")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--title", type=str, default=None)

    # Aggregate-mode flags
    p.add_argument("--noise-compare", action="store_true",
                   help="generate noise-level comparison plot (requires "
                   "aggregated CSV with --dataset and --arch)")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--arch", type=str, default=None,
                   help="architecture string, e.g. '[25, 25, 25]'")
    p.add_argument("--layer", type=int, default=None,
                   help="layer for aggregate plots (default: deepest)")

    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.csv.exists():
        print(f"missing CSV: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")

    if args.noise_compare:
        if args.dataset is None or args.arch is None:
            print("--noise-compare requires --dataset and --arch", file=sys.stderr)
            return 2
        sub = df[(df["dataset"] == args.dataset) & (df["arch_str"] == args.arch)]
        if sub.empty:
            print(f"no rows for dataset={args.dataset!r}, arch={args.arch!r}", file=sys.stderr)
            return 3
        layer = args.layer if args.layer is not None else int(sub["layer"].max())
        title = (args.title
                 or f"{args.dataset} {args.arch} | ε={args.epsilon} | layer={layer}")
        arch_tag = args.arch.replace(" ", "").replace("[", "").replace("]", "").replace(",", "x")
        outpath = (
            args.outdir
            / f"noise_compare_{args.dataset}_a{arch_tag}_L{layer}_eps{args.epsilon}.png"
        )
        plot_noise_compare(sub, args.epsilon, layer, title, outpath)
        return 0

    title = args.title or f"{args.csv.parent.name} | {args.csv.stem} | ε={args.epsilon}"
    outpath = (
        args.outdir
        / f"{args.csv.parent.name}__{args.csv.stem}_dashboard_eps{args.epsilon}.png"
    )
    make_dashboard(df, args.epsilon, args.collapse_epoch, title, outpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
