"""
Closed-form / hand-computed gate for the Phase A.2 gen-gap predictors.

Each check uses a hand-constructed model whose Frobenius / path-norm /
spectral-margin / sharpness value is known analytically, then asserts the
implementation matches to floating-point tolerance. Run before the full
sweep:

    uv run python scripts/validate_gen_gap_predictors.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src_experiment.baselines.gen_gap_predictors import (  # noqa: E402
    frobenius,
    path_norm,
    sharpness,
    spectral_margin,
)
from src_experiment.utils import NeuralNet  # noqa: E402


def _set_weights(model: NeuralNet, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
    """Overwrite each l_i with the supplied (W, b)."""
    n_layers = len(model.hidden_sizes) + 1
    assert len(weights) == n_layers and len(biases) == n_layers
    with torch.no_grad():
        for i, (W, b) in enumerate(zip(weights, biases), start=1):
            layer = getattr(model, f"l{i}")
            layer.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
            layer.bias.copy_(torch.as_tensor(b, dtype=torch.float32))


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
            for n, d in self.failures:
                print(f"  - {n}: {d}")
            return 1
        print("\nALL TESTS PASSED")
        return 0


def main() -> int:
    rep = _Reporter()

    # ---- (1) Frobenius on identity --------------------------------------
    print("\n[1] Frobenius on identity weights (target: sqrt(rank sums))")
    d = 5
    model = NeuralNet(input_size=d, hidden_sizes=[d], num_classes=d, dropout=0.0)
    _set_weights(
        model,
        weights=[np.eye(d), np.eye(d)],
        biases=[np.zeros(d), np.zeros(d)],
    )
    out = frobenius(model)
    truth = float(np.sqrt(d + d))  # sum of two identity Frobenius² = 2d
    rep.check(
        "frobenius identity",
        abs(out["frobenius"] - truth) < 1e-5,
        f"truth={truth:.4f}  est={out['frobenius']:.4f}",
    )

    # ---- (2) Path-norm on 2-layer all-ones model ------------------------
    print("\n[2] Path-norm on a 2-layer all-ones model (closed form)")
    # input_size=2 → hidden=3 → output=2. All weights = 1, biases = 0.
    # Number of edges per path = 2 (one per layer).
    # |paths| = 2 (input units) * 3 (hidden) * 2 (output) = 12 paths.
    # Each path has product of squared edges = 1. So squared path-norm = 12.
    in_dim, hid_dim, out_dim = 2, 3, 2
    model = NeuralNet(input_size=in_dim, hidden_sizes=[hid_dim], num_classes=out_dim, dropout=0.0)
    _set_weights(
        model,
        weights=[np.ones((hid_dim, in_dim)), np.ones((out_dim, hid_dim))],
        biases=[np.zeros(hid_dim), np.zeros(out_dim)],
    )
    out = path_norm(model)
    truth = float(np.sqrt(in_dim * hid_dim * out_dim))  # = sqrt(12)
    rep.check(
        "path_norm all-ones",
        abs(out["path_norm"] - truth) < 1e-5,
        f"truth={truth:.4f}  est={out['path_norm']:.4f}",
    )

    # ---- (3) Spectral product on identity weights -----------------------
    print("\n[3] Spectral product on identity weights (target: 1.0)")
    d = 4
    model = NeuralNet(input_size=d, hidden_sizes=[d], num_classes=d, dropout=0.0)
    _set_weights(
        model,
        weights=[np.eye(d), np.eye(d)],
        biases=[np.zeros(d), np.zeros(d)],
    )
    # Use one-hot inputs so post-ReLU = input and logits = X — known margins.
    X = np.eye(d, dtype=np.float32)
    Y = np.arange(d, dtype=np.int64)
    out = spectral_margin(model, X, Y)
    rep.check(
        "spectral_prod=1",
        abs(out["spectral_prod"] - 1.0) < 1e-5,
        f"spectral_prod={out['spectral_prod']:.6f}",
    )
    rep.check(
        "gamma_min=1 (identity logits, one-hot inputs)",
        abs(out["gamma_min"] - 1.0) < 1e-5,
        f"gamma_min={out['gamma_min']:.6f}",
    )

    # ---- (4) Sharpness — match scipy eigsh against full Hessian ---------
    print("\n[4] Sharpness — top-5 eigenvalues match brute-force Hessian")
    # Build a tiny network so brute-force full Hessian is cheap.
    in_dim, hid_dim, out_dim, N = 3, 4, 3, 32
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, in_dim)).astype(np.float32)
    Y = rng.integers(0, out_dim, size=N).astype(np.int64)
    model = NeuralNet(input_size=in_dim, hidden_sizes=[hid_dim], num_classes=out_dim,
                      dropout=0.0, seed=42)
    sh = sharpness(model, X, Y, k=5, n_subsample=N, seed=0)

    # Brute-force full Hessian for ground truth.
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    criterion = torch.nn.CrossEntropyLoss()
    X_t = torch.as_tensor(X)
    Y_t = torch.as_tensor(Y)

    def hvp_brute(v_np: np.ndarray) -> np.ndarray:
        v = torch.as_tensor(v_np.astype(np.float32))
        v_split = []
        off = 0
        for p in params:
            n = p.numel()
            v_split.append(v[off : off + n].view_as(p))
            off += n
        loss = criterion(model(X_t), Y_t)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * vp).sum() for g, vp in zip(grads, v_split))
        Hv = torch.autograd.grad(dot, params)
        return torch.cat([h.contiguous().view(-1) for h in Hv]).detach().numpy().astype(np.float64)

    H = np.zeros((n_params, n_params))
    for i in range(n_params):
        e = np.zeros(n_params)
        e[i] = 1.0
        H[:, i] = hvp_brute(e)
    H = 0.5 * (H + H.T)  # symmetrise (numerical)
    eigs_full = np.linalg.eigvalsh(H)
    truth_lambda_max = float(eigs_full.max())
    truth_top5_sum = float(np.sort(eigs_full)[-5:].sum())

    rep.check(
        "sharpness lambda_max",
        abs(sh["lambda_max"] - truth_lambda_max) < 1e-3,
        f"truth={truth_lambda_max:.4f}  est={sh['lambda_max']:.4f}",
    )
    rep.check(
        "sharpness lambda_sum_top5",
        abs(sh["lambda_sum_top"] - truth_top5_sum) < 1e-2,
        f"truth={truth_top5_sum:.4f}  est={sh['lambda_sum_top']:.4f}",
    )
    rep.check("sharpness Lanczos converged", sh["converged"], f"converged={sh['converged']}")

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
