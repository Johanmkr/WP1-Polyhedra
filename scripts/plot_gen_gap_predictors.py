"""Phase A.2 — Generalization-gap predictor comparison plots.

Reads `results/gen_gap_predictors.csv` (per-HDF5 sharpness, path-norm,
Frobenius, spectral-margin) and joins `rho_func` / `bits_ours_func` /
`bits_ours_raw` from the per-dataset `*_label_noise_new_estimator.csv`
files at the **same slice as Exp 2** (deepest layer, ε = 10, last epoch
per (arch, seed)). The `rho_func` column inside `mi_baselines.csv` is
recorded at ε = 0.0001 (degenerate "no merging") and is not the right
quantity to rank against gen-gap — see Exp 2.

Sign convention: positive Kendall τ = "predictor↑ tracks gen_gap_acc↑",
i.e. larger predictor value matches worse generalization. For predictors
whose natural direction is the opposite (spectral_margin_ratio,
bits_ours_raw — both larger-is-better), we flip the sign before
correlating so a positive bar always means "good predictor".

Outputs:
- `figures/baseline_gen_gap_kendall.png` — two-row figure per dataset.
  Row 1: cross-cell Kendall τ bar with 95% bootstrap CI per predictor.
  Row 2: within-noise τ as a small bar group per noise level.
- `results/gen_gap_predictors_kendall.csv` — one row per
  (dataset, subset, predictor): τ, lo, hi, n_cells, sign_flipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260428)
N_BOOT = 2000
EXP2_EPS = 10.0
ESTIMATOR_CSVS = {
    "composite": "composite_label_noise_new_estimator.csv",
    "wbc":       "wbc_label_noise_new_estimator.csv",
}

# (csv column → display label, flip_sign).
# flip_sign = True if the predictor is naturally "larger = better",
# so we negate it so positive τ means "good predictor".
PREDICTORS = [
    ("frobenius",              "Frobenius",       False),
    ("log_path_norm",          "log path-norm",   False),
    ("lambda_max_median",      "sharpness λ_max", False),
    ("spectral_margin_ratio",  "spectral margin", True),
    ("rho_func",               "ρ_func (ours)",   False),
    ("rl_proxy",               "rl_proxy (ours)", False),
    ("bits_ours_raw",          "Ĩ_raw (ours)",    True),
]
PREDICTOR_COLORS = {
    "Frobenius":         "#1f77b4",
    "log path-norm":     "#2ca02c",
    "sharpness λ_max":   "#8c564b",
    "spectral margin":   "#e377c2",
    "ρ_func (ours)":     "#d62728",
    "rl_proxy (ours)":   "#ff7f0e",
    "Ĩ_raw (ours)":      "#9467bd",
}


def _exp2_slice(dataset: str) -> pd.DataFrame:
    """Deepest layer × ε=10 × last epoch per (arch, seed), matching Exp 2."""
    path = REPO / "results" / ESTIMATOR_CSVS[dataset]
    df = pd.read_csv(path)
    deepest = df.groupby("arch_str")["layer"].transform("max")
    sub = df[(df["layer"] == deepest) & (np.isclose(df["epsilon"], EXP2_EPS))].copy()
    last_ep = sub.groupby(["arch_str", "seed"])["epoch"].transform("max")
    sub = sub[sub["epoch"] == last_ep].copy()
    sub["dataset"] = dataset
    bits_raw = sub["plug_in_bits"] + sub["miller_madow_bits"]
    bits_func = sub["plug_in_func_bits"] + sub["miller_madow_func_bits"]
    return sub.assign(bits_ours_raw=bits_raw, bits_ours_func=bits_func)[[
        "dataset", "noise_level", "arch_str", "seed",
        "rho", "rho_func", "rl_proxy",
        "bits_ours_raw", "bits_ours_func",
    ]]


def load() -> pd.DataFrame:
    gg = pd.read_csv(REPO / "results" / "gen_gap_predictors.csv")
    exp2 = pd.concat(
        [_exp2_slice(d) for d in ESTIMATOR_CSVS], ignore_index=True,
    )
    keys = ["dataset", "noise_level", "arch_str", "seed"]
    df = gg.merge(exp2, on=keys, how="left")
    missing = df["rho_func"].isna().sum()
    if missing:
        print(f"warning: {missing} rows missing rho_func @ ε={EXP2_EPS}")
    return df


def kendall_with_ci(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"tau": np.nan, "lo": np.nan, "hi": np.nan, "n": int(len(x))}
    tau = kendalltau(x, y).statistic
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        xi, yi = x[idx[i]], y[idx[i]]
        if np.std(xi) == 0 or np.std(yi) == 0:
            boot[i] = np.nan
            continue
        boot[i] = kendalltau(xi, yi).statistic
    return {
        "tau": float(tau),
        "lo": float(np.nanpercentile(boot, 2.5)),
        "hi": float(np.nanpercentile(boot, 97.5)),
        "n":  int(len(x)),
    }


def compute_taus(df: pd.DataFrame, dataset: str,
                 noise: float | None) -> pd.DataFrame:
    sub = df[df["dataset"] == dataset]
    if noise is not None:
        sub = sub[sub["noise_level"] == noise]
    rows: list[dict] = []
    y = sub["gen_gap_acc"].to_numpy()
    for col, label, flip in PREDICTORS:
        if col not in sub.columns:
            continue
        x = sub[col].to_numpy()
        if flip:
            x = -x
        stats = kendall_with_ci(x, y)
        rows.append({
            "dataset": dataset,
            "subset": "all" if noise is None else f"noise={noise}",
            "predictor": label,
            "predictor_col": col,
            "sign_flipped": flip,
            **stats,
        })
    return pd.DataFrame(rows)


def plot_dataset_row(ax_main, ax_within, df: pd.DataFrame,
                     dataset: str, noises: list[float]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    cross = compute_taus(df, dataset, None).sort_values("tau", ascending=False)
    rows.append(cross)

    # Cross-cell bars.
    xs = np.arange(len(cross))
    colors = [PREDICTOR_COLORS[p] for p in cross["predictor"]]
    ax_main.bar(xs, cross["tau"], color=colors, edgecolor="black",
                linewidth=0.5)
    ax_main.errorbar(xs, cross["tau"],
                     yerr=[cross["tau"] - cross["lo"],
                           cross["hi"] - cross["tau"]],
                     fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    for i, (tau, n) in enumerate(zip(cross["tau"], cross["n"])):
        ax_main.text(i, tau + (0.02 if tau >= 0 else -0.04),
                     f"{tau:+.2f}\nn={n}", ha="center",
                     va="bottom" if tau >= 0 else "top",
                     fontsize=7)
    ax_main.set_xticks(xs)
    ax_main.set_xticklabels(cross["predictor"], rotation=20, ha="right",
                            fontsize=8)
    ax_main.axhline(0, color="black", lw=0.6)
    ax_main.set_ylabel("Kendall τ vs gen_gap_acc")
    ax_main.set_title(f"{dataset}: cross-cell τ (95% bootstrap CI)",
                      fontsize=10)
    ax_main.set_ylim(-1.0, 1.0)
    ax_main.grid(alpha=0.3, axis="y")

    # Within-noise grouped bars.
    within = pd.concat(
        [compute_taus(df, dataset, n) for n in noises], ignore_index=True,
    )
    rows.append(within)
    predictors = list(cross["predictor"])
    width = 0.8 / max(len(noises), 1)
    base_x = np.arange(len(predictors))
    for i, n in enumerate(noises):
        ws = within[within["subset"] == f"noise={n}"].set_index("predictor")
        ws = ws.reindex(predictors)
        offset = (i - (len(noises) - 1) / 2) * width
        ax_within.bar(base_x + offset, ws["tau"], width=width,
                      label=f"n={n}",
                      color=plt.cm.viridis(0.15 + 0.7 * i / max(len(noises) - 1, 1)),
                      edgecolor="black", linewidth=0.4)
        ax_within.errorbar(base_x + offset, ws["tau"],
                           yerr=[ws["tau"] - ws["lo"], ws["hi"] - ws["tau"]],
                           fmt="none", ecolor="black", elinewidth=0.5,
                           capsize=1.5)
    ax_within.set_xticks(base_x)
    ax_within.set_xticklabels(predictors, rotation=20, ha="right", fontsize=8)
    ax_within.axhline(0, color="black", lw=0.6)
    ax_within.set_ylabel("Kendall τ")
    ax_within.set_title(f"{dataset}: within-noise τ (controls noise confound)",
                        fontsize=10)
    ax_within.set_ylim(-1.0, 1.0)
    ax_within.grid(alpha=0.3, axis="y")
    ax_within.legend(fontsize=7, ncol=len(noises))

    return pd.concat(rows, ignore_index=True)


def main() -> int:
    df = load()
    datasets = sorted(df["dataset"].unique())
    noises = sorted(df["noise_level"].unique())

    n_ds = len(datasets)
    fig, axes = plt.subplots(
        2, n_ds, figsize=(5.8 * n_ds, 9.0),
        gridspec_kw=dict(hspace=0.55, wspace=0.30,
                         left=0.07, right=0.98, top=0.93, bottom=0.10),
    )
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    summary_frames: list[pd.DataFrame] = []
    for j, dataset in enumerate(datasets):
        s = plot_dataset_row(axes[0, j], axes[1, j], df, dataset, noises)
        summary_frames.append(s)

    fig.suptitle(
        "Phase A.2 — generalization-gap predictors. "
        "Positive τ = predictor↑ tracks gen_gap_acc↑. "
        "Sign-flipped predictors (spectral margin, Ĩ_raw) negated so "
        "positive = good.",
        fontsize=11, y=0.99,
    )

    figpath = REPO / "figures" / "baseline_gen_gap_kendall.png"
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figpath}")

    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = REPO / "results" / "gen_gap_predictors_kendall.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    print("\nKendall τ vs gen_gap_acc:")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
