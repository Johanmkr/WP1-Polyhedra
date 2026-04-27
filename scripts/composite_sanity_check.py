"""Sanity checks + headline-number extraction for the composite label-noise sweep.

Run after `--aggregate` produces results/composite_label_noise_new_estimator.csv.
Mirrors the checklist in planning/next_phase.md §4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("results/composite_label_noise_new_estimator.csv")
EPS = 10.0


def main() -> int:
    if not CSV.exists():
        print(f"missing {CSV}; run --aggregate first")
        return 1

    df = pd.read_csv(CSV)
    print(f"loaded {CSV}: shape={df.shape}")
    print(f"archs: {sorted(df['arch_str'].unique())}")
    print(f"noise levels: {sorted(df['noise_level'].unique())}")
    print(f"seeds: {sorted(df['seed'].unique())}")
    print(f"epochs: {sorted(df['epoch'].unique())}")
    print()

    # 1. DPI per row
    dpi_ok = (df["plug_in_func_bits"] <= df["plug_in_bits"] + 1e-9).all()
    n_dpi_violations = (~(df["plug_in_func_bits"] <= df["plug_in_bits"] + 1e-9)).sum()
    print(f"[DPI] plug_in_func ≤ plug_in everywhere: {dpi_ok} ({n_dpi_violations} violations)")
    dpi_mm_ok = (df["miller_madow_func_bits"] <= df["miller_madow_bits"] + 1e-9).all()
    n_mm_v = (~(df["miller_madow_func_bits"] <= df["miller_madow_bits"] + 1e-9)).sum()
    print(f"[DPI] miller_madow_func ≤ miller_madow everywhere: {dpi_mm_ok} ({n_mm_v} violations)")
    print()

    # 2. trustworthiness map
    print("[trustworthiness] max ρ per arch (across all epochs/layers/noise)")
    rho_max = df.groupby("arch_str")["rho"].max().sort_values()
    for a, v in rho_max.items():
        flag = "✓" if v < 0.3 else ("borderline" if v < 0.5 else "✗")
        print(f"  {a:<28s} max ρ = {v:.3f}  {flag}")
    print()

    # 3. truncation_prob non-null
    n_null = df["truncation_prob"].isna().sum()
    print(f"[truncation_prob] non-null in all rows: {n_null == 0} (NaN count = {n_null})")
    if n_null < len(df):
        tp = df["truncation_prob"].dropna()
        print(f"  truncation_prob: mean={tp.mean():.4f}, median={tp.median():.4f}, max={tp.max():.4f}")
    print()

    # 4. headline I(Y;Ω) and I_func(Y;Ω) at deepest layer, ε=10, last epoch, mean over seeds
    print(f"[headline] I_raw and I_func at deepest layer, ε={EPS}, last epoch (mean ± std over seeds)")
    last_epoch = df["epoch"].max()
    sub = df[(df["epoch"] == last_epoch) & (np.isclose(df["epsilon"], EPS))]

    rows = []
    for arch, g in sub.groupby("arch_str"):
        deepest = g["layer"].max()
        gd = g[g["layer"] == deepest]
        i_raw_mean = gd["miller_madow_bits"].mean()
        i_raw_std = gd["miller_madow_bits"].std()
        i_func_mean = gd["miller_madow_func_bits"].mean()
        i_func_std = gd["miller_madow_func_bits"].std()
        # noise comparison
        noise_grp = gd.groupby("noise_level")
        per_noise = {
            n: {
                "i_raw": noise_grp.get_group(n)["miller_madow_bits"].mean() if n in noise_grp.groups else np.nan,
                "i_func": noise_grp.get_group(n)["miller_madow_func_bits"].mean() if n in noise_grp.groups else np.nan,
            }
            for n in [0.0, 0.2, 0.4]
        }
        rows.append((arch, deepest, per_noise))

    print(f"{'arch':<26s} L  metric  n=0.0    n=0.2    n=0.4")
    for arch, L, pn in rows:
        ir = (pn[0.0]["i_raw"], pn[0.2]["i_raw"], pn[0.4]["i_raw"])
        iq = (pn[0.0]["i_func"], pn[0.2]["i_func"], pn[0.4]["i_func"])
        print(f"{arch:<26s} {L}  I_raw   {ir[0]:7.3f}  {ir[1]:7.3f}  {ir[2]:7.3f}")
        print(f"{arch:<26s} {L}  I_func  {iq[0]:7.3f}  {iq[1]:7.3f}  {iq[2]:7.3f}")
    print()

    # 5. Quotient-vs-raw noise sensitivity table
    print(f"[noise drop] I(n=0) − I(n=0.4), deepest layer, ε={EPS}, last epoch")
    print(f"{'arch':<26s} L   raw drop   quotient drop")
    for arch, L, pn in rows:
        raw_drop = pn[0.0]["i_raw"] - pn[0.4]["i_raw"]
        q_drop = pn[0.0]["i_func"] - pn[0.4]["i_func"]
        marker = "★" if q_drop > raw_drop else " "
        print(f"{arch:<26s} {L}  {raw_drop:8.3f}   {q_drop:8.3f}   {marker}")
    print()

    # 6. plateau location
    print(f"[plateau] median ρ_func vs ε at deepest layer, last epoch, mean over noise+seeds")
    pl = df[(df["epoch"] == last_epoch)].groupby(["arch_str", "epsilon"])["rho_func"].mean().reset_index()
    for arch, gd in pl.groupby("arch_str"):
        deepest = df[df["arch_str"] == arch]["layer"].max()
        gdd = df[(df["arch_str"] == arch) & (df["layer"] == deepest) & (df["epoch"] == last_epoch)]
        per_eps = gdd.groupby("epsilon")["rho_func"].mean()
        print(f"  {arch:<26s}  ", " ".join(f"ε={e:6.4g}:ρf={v:.3f}" for e, v in per_eps.items()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
