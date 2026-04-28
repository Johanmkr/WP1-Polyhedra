"""
Synthetic ground-truth gate for the Phase A.1 MI baselines.

Runs the closed-form / MC-truth checks described in
``planning/phase_a_baselines.md`` §A.1.4. Exits 0 if every assertion passes,
1 otherwise. Run before any full sweep:

    uv run python scripts/validate_mi_baselines.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src_experiment.baselines.mi_baselines import (  # noqa: E402
    InfoNCEEstimator,
    MINEEstimator,
    binning_mi,
    kmeans_mi,
    ksg_mi,
)


# ---------------------------------------------------------------------------
# Closed-form / MC truths
# ---------------------------------------------------------------------------
def truth_discrete(K: int, eps: float) -> float:
    """I(X;Y) for X uniform on {0,…,K-1}, Y = X w.p. 1-ε else uniform."""
    p = np.full(K, eps / K)
    p[0] += 1 - eps
    H = -(p[p > 0] * np.log2(p[p > 0])).sum()
    return float(np.log2(K) - H)


def truth_mixed_gaussian(mu: float, d: int = 1, n_grid: int = 1_000_000) -> float:
    """I(T;Y) for Y ∈ {0,1}, T|Y ~ N(±μ·1, I_d). Numerical via large MC."""
    rng = np.random.default_rng(123)
    Y = rng.integers(0, 2, size=n_grid)
    T = rng.standard_normal(size=(n_grid, d)) + np.where(Y[:, None] == 1, mu, -mu)
    log_p_T_y0 = -0.5 * np.sum((T + mu) ** 2, axis=1)
    log_p_T_y1 = -0.5 * np.sum((T - mu) ** 2, axis=1)
    log_p_T_given_y = np.where(Y == 1, log_p_T_y1, log_p_T_y0)
    log_p_T = np.logaddexp(log_p_T_y0, log_p_T_y1) - np.log(2.0)
    return float((log_p_T_given_y - log_p_T).mean() / np.log(2))


def truth_gaussian_sign(rho: float) -> float:
    """I(X1; sign(X2)) for (X1, X2) ~ N(0, [[1,ρ],[ρ,1]])."""
    from scipy.stats import norm

    rng = np.random.default_rng(7)
    X = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=2_000_000)
    p1 = norm.cdf(rho * X[:, 0] / np.sqrt(1 - rho ** 2))
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    HYX = -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1)).mean()
    return float(1.0 - HYX)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
class _Reporter:
    def __init__(self) -> None:
        self.failures: list[Tuple[str, str]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not condition:
            self.failures.append((name, detail))

    def finish(self) -> int:
        if self.failures:
            print(f"\n{len(self.failures)} FAIL(S):")
            for name, detail in self.failures:
                print(f"  - {name}: {detail}")
            return 1
        print("\nALL TESTS PASSED")
        return 0


def main() -> int:
    rep = _Reporter()
    rng = np.random.default_rng(0)

    # ---- (1) discrete I(X;Y): binning + kmeans ---------------------------
    print("\n[1] discrete I(X;Y): binning + kmeans (target ±0.05 bits)")
    K, eps, N = 4, 0.2, 10_000
    X = rng.integers(0, K, size=N)
    Y = np.where(rng.random(N) < eps, rng.integers(0, K, size=N), X)
    T = (2.0 * X / (K - 1) - 1.0).reshape(-1, 1).astype(np.float32)
    mi_true = truth_discrete(K, eps)

    out = binning_mi(T, Y, n_bins=K)
    rep.check("binning", abs(out["bits"] - mi_true) < 0.05,
              f"truth={mi_true:.3f}  est={out['bits']:.3f}  Δ={abs(out['bits']-mi_true):.3f}")

    out = kmeans_mi(T, Y, K=K, seed=0)
    rep.check("kmeans", abs(out["bits"] - mi_true) < 0.05,
              f"truth={mi_true:.3f}  est={out['bits']:.3f}  Δ={abs(out['bits']-mi_true):.3f}")

    # ---- (2) mixed Gaussian: KSG -----------------------------------------
    print("\n[2] mixed Gaussian-discrete I(T;Y): KSG-Ross (target ±0.05 bits, slack 0.10)")
    for d in (1, 5):
        for mu in (0.5, 1.0, 2.0):
            N_k = 5_000
            Y_k = rng.integers(0, 2, size=N_k)
            T_k = rng.standard_normal(size=(N_k, d)) + np.where(Y_k[:, None] == 1, mu, -mu)
            truth = truth_mixed_gaussian(mu, d=d)
            out = ksg_mi(T_k, Y_k, k=3)
            delta = abs(out["bits"] - truth)
            rep.check(
                f"ksg d={d} mu={mu}",
                delta < 0.10,
                f"truth={truth:.3f}  est={out['bits']:.3f}  Δ={delta:.3f}  wall={out['wall']:.2f}s",
            )

    # ---- (3) Gaussian-sign: MINE / InfoNCE -------------------------------
    print("\n[3] Gaussian-sign I(X1; sign(X2)): MINE (±0.10 bits) and InfoNCE (±0.20 bits)")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    using torch device={device}")
    for rho in (0.5, 0.8):
        N_g = 5_000
        X = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=N_g).astype(np.float32)
        T_g = X[:, :1]
        Y_g = (X[:, 1] > 0).astype(np.int64)
        truth = truth_gaussian_sign(rho)

        t0 = time.perf_counter()
        nce = InfoNCEEstimator(n_iter=1500, batch=256, device=device).estimate(
            T_g, Y_g, num_classes=2, seed=0
        )
        rep.check(
            f"infonce ρ={rho}",
            abs(nce["bits"] - truth) < 0.20,
            f"truth={truth:.3f}  est={nce['bits']:.3f}  Δ={abs(nce['bits']-truth):.3f}  wall={nce['wall']:.1f}s",
        )

        seeds = [
            MINEEstimator(n_iter=2500, device=device).estimate(
                T_g, Y_g, num_classes=2, seed=s
            )["bits"]
            for s in range(3)
        ]
        m_mean, m_std = float(np.mean(seeds)), float(np.std(seeds))
        rep.check(
            f"mine ρ={rho}",
            abs(m_mean - truth) < 0.10,
            f"truth={truth:.3f}  est={m_mean:.3f}±{m_std:.3f}  Δ={abs(m_mean-truth):.3f}  wall(per seed)≈{(time.perf_counter()-t0-nce['wall'])/3:.1f}s",
        )

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
