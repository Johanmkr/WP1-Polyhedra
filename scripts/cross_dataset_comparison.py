"""Cross-dataset noise-drop comparison: composite vs WBC.

Pulls shared archs at the deepest *trustworthy* layer, computes raw and quotient
MI drop (n=0 → n=0.4) at ε=10, last epoch, mean over seeds, and prints a side-by-side
table. Optionally writes a comparison figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WBC_CSV = Path("results/wbc_label_noise_new_estimator.csv")
COMP_CSV = Path("results/composite_label_noise_new_estimator.csv")
SHARED_ARCHS = ["[5, 5, 5]", "[7, 7, 7]", "[25, 25, 25]"]
EPS = 10.0
RHO_THRESHOLD = 0.3


def deepest_trustworthy_layer(df: pd.DataFrame, arch: str) -> int:
    """Return the deepest layer for which max ρ stays below the trust threshold."""
    sub = df[df["arch_str"] == arch]
    if sub.empty:
        return -1
    layer_rho = sub.groupby("layer")["rho"].max()
    trust = layer_rho[layer_rho < RHO_THRESHOLD]
    if trust.empty:
        # fall back to deepest layer with min ρ; we'll still report (caveat)
        return int(sub["layer"].max())
    return int(trust.index.max())


def headline(df: pd.DataFrame, arch: str, layer: int) -> dict:
    last_epoch = df["epoch"].max()
    sub = df[
        (df["arch_str"] == arch)
        & (df["layer"] == layer)
        & (df["epoch"] == last_epoch)
        & (np.isclose(df["epsilon"], EPS))
    ]
    out = {}
    for n in [0.0, 0.2, 0.4]:
        g = sub[np.isclose(sub["noise_level"], n)]
        out[("i_raw", n)] = g["miller_madow_bits"].mean()
        out[("i_func", n)] = g["miller_madow_func_bits"].mean()
    out["raw_drop"] = out[("i_raw", 0.0)] - out[("i_raw", 0.4)]
    out["q_drop"] = out[("i_func", 0.0)] - out[("i_func", 0.4)]
    out["max_rho"] = sub["rho"].max()
    return out


def main() -> int:
    if not COMP_CSV.exists():
        print(f"missing {COMP_CSV}; cannot compare yet")
        return 1
    if not WBC_CSV.exists():
        print(f"missing {WBC_CSV}; cannot compare")
        return 1

    df_wbc = pd.read_csv(WBC_CSV)
    df_comp = pd.read_csv(COMP_CSV)

    rows = []
    for arch in SHARED_ARCHS:
        l_wbc = deepest_trustworthy_layer(df_wbc, arch)
        l_comp = deepest_trustworthy_layer(df_comp, arch)
        h_wbc = headline(df_wbc, arch, l_wbc)
        h_comp = headline(df_comp, arch, l_comp)
        rows.append({
            "arch": arch,
            "wbc_L": l_wbc,
            "wbc_max_rho": h_wbc["max_rho"],
            "wbc_raw_drop": h_wbc["raw_drop"],
            "wbc_q_drop": h_wbc["q_drop"],
            "comp_L": l_comp,
            "comp_max_rho": h_comp["max_rho"],
            "comp_raw_drop": h_comp["raw_drop"],
            "comp_q_drop": h_comp["q_drop"],
        })

    print(f"{'arch':<14s}  {'WBC':<28s}  {'composite':<28s}")
    print(f"{'':<14s}  {'L  raw_drop  q_drop  ρmax':<28s}  {'L  raw_drop  q_drop  ρmax':<28s}")
    for r in rows:
        wbc = f"{r['wbc_L']}  {r['wbc_raw_drop']:7.3f}  {r['wbc_q_drop']:7.3f}  {r['wbc_max_rho']:.3f}"
        comp = f"{r['comp_L']}  {r['comp_raw_drop']:7.3f}  {r['comp_q_drop']:7.3f}  {r['comp_max_rho']:.3f}"
        print(f"{r['arch']:<14s}  {wbc:<28s}  {comp:<28s}")

    # side-by-side bar chart
    outdir = Path("figures/label_noise_new_estimator/cross_dataset")
    outdir.mkdir(parents=True, exist_ok=True)
    archs = [r["arch"] for r in rows]
    wbc_raw = [r["wbc_raw_drop"] for r in rows]
    wbc_q = [r["wbc_q_drop"] for r in rows]
    comp_raw = [r["comp_raw_drop"] for r in rows]
    comp_q = [r["comp_q_drop"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    x = np.arange(len(archs))
    width = 0.35

    axes[0].bar(x - width / 2, wbc_raw, width, label="raw I_raw")
    axes[0].bar(x + width / 2, wbc_q, width, label="quotient I_func")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(archs, rotation=20, ha="right")
    axes[0].set_ylabel("noise drop  I(n=0) − I(n=0.4)  [bits]")
    axes[0].set_title("WBC")
    axes[0].legend()
    axes[0].axhline(0, color="k", linewidth=0.5)

    axes[1].bar(x - width / 2, comp_raw, width, label="raw I_raw")
    axes[1].bar(x + width / 2, comp_q, width, label="quotient I_func")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(archs, rotation=20, ha="right")
    axes[1].set_title("composite")
    axes[1].legend()
    axes[1].axhline(0, color="k", linewidth=0.5)

    fig.suptitle(f"Noise sensitivity: I(n=0) − I(n=0.4)  [ε={EPS}, deepest trustworthy layer]")
    fig.tight_layout()
    out_path = outdir / "noise_drop_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")

    # verdict
    n_q_wins = sum(1 for r in rows if r["comp_q_drop"] > r["comp_raw_drop"])
    print(f"\nquotient drop > raw drop in {n_q_wins}/{len(rows)} composite archs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
