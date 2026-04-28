"""
Phase A.2 generalization-gap predictors: standard scalar measures we
benchmark our `rho_func` / `rl_proxy` against under the Kendall-τ protocol
of Jiang et al. 2020. See ``planning/phase_a_baselines.md`` §A.2.

Predictors implemented (per A.2.1):
  - :func:`frobenius`        — Σ_l ||W_l||_F  (trivial scale baseline).
  - :func:`path_norm`        — sqrt(Σ_paths Π_l w_e²) via squared-weight forward.
  - :func:`spectral_margin`  — γ_min(X, Y) / Π_l ||W_l||_2  (Bartlett 2017,
                               Jiang 2020 simplified).
  - :func:`sharpness`        — top-k Hessian eigenvalues of CE loss at θ via
                               Lanczos on Hessian-vector products
                               (Keskar 2017, Foret 2021).

The reconstruction helper :func:`load_neural_net_from_h5` rebuilds a
``src_experiment.utils.NeuralNet`` from a saved HDF5 epoch so the same
PyTorch graph used at training time backs the HVP for sharpness.

Note on data
------------
All predictors that depend on data (sharpness, spectral_margin) accept a
generic ``(X, Y)`` probe rather than the training set. Using a probe in
the training-time feature space is a deliberate, documented deviation
from the strict Jiang 2020 prescription: replaying the noise-injected
training pipeline per-HDF5 is expensive, and Kendall τ ranks are robust
to using a held-out probe instead. Tests of train-vs-probe sharpness
agreement can be added later if a referee asks for them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_neural_net_from_h5(h5_path: PathLike, epoch: int, device: str = "cpu"):
    """Rebuild a :class:`NeuralNet` with the weights saved at ``epoch``.

    Reads architecture / input_size / num_classes from ``metadata`` attrs,
    constructs a fresh :class:`NeuralNet` (dropout disabled — we only ever
    eval), and loads ``l<i>.weight`` / ``l<i>.bias`` from the HDF5.
    """
    import torch

    from src_experiment.utils import NeuralNet

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["metadata"].attrs)
        architecture = [int(x) for x in attrs["architecture"]]
        input_size = int(attrs["inferred_input_size"])
        num_classes = int(attrs["inferred_num_classes"])

    model = NeuralNet(
        input_size=input_size,
        hidden_sizes=architecture,
        num_classes=num_classes,
        dropout=0.0,
    ).to(device)
    model.eval()

    n_layers = len(architecture) + 1  # incl. output
    with h5py.File(h5_path, "r") as f:
        grp = f[f"epochs/epoch_{epoch}"]
        for i in range(1, n_layers + 1):
            W = np.asarray(grp[f"l{i}.weight"][:], dtype=np.float32)
            b = np.asarray(grp[f"l{i}.bias"][:], dtype=np.float32)
            layer = getattr(model, f"l{i}")
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(W))
                layer.bias.copy_(torch.from_numpy(b))
    return model


def _layer_weights(model) -> List[np.ndarray]:
    """Return [W_1, …, W_{L+1}] as float64 numpy arrays."""
    out: List[np.ndarray] = []
    n_layers = len(model.hidden_sizes) + 1
    for i in range(1, n_layers + 1):
        out.append(getattr(model, f"l{i}").weight.detach().cpu().numpy().astype(np.float64))
    return out


# ---------------------------------------------------------------------------
# Frobenius
# ---------------------------------------------------------------------------
def frobenius(model) -> Dict[str, float]:
    """Total Frobenius norm ||W||_F = sqrt(Σ_l Σ_ij w²)."""
    t0 = time.perf_counter()
    sq = sum(float((W ** 2).sum()) for W in _layer_weights(model))
    return {"frobenius": float(np.sqrt(sq)), "wall": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# Path-norm
# ---------------------------------------------------------------------------
def path_norm(model) -> Dict[str, float]:
    """L² path-norm via the squared-weight forward trick.

    ``||W||_path² = Σ_paths Π_e w_e²`` is the sum of squared-weight products
    along all input→output paths. Equivalently, with each layer's weights
    elementwise-squared, biases zeroed, ReLU dropped, and a 1-vector input,
    the network's output has elements summing to the squared path-norm.
    """
    t0 = time.perf_counter()
    Ws = _layer_weights(model)
    a = np.ones(Ws[0].shape[1], dtype=np.float64)
    for W in Ws:
        a = (W ** 2) @ a  # no bias, no ReLU
    sq_path = float(a.sum())
    return {
        "path_norm": float(np.sqrt(sq_path)),
        "log_path_norm": float(0.5 * np.log(sq_path)) if sq_path > 0 else float("-inf"),
        "wall": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Spectral margin
# ---------------------------------------------------------------------------
def spectral_margin(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    """γ_min(X, Y) / Π_l ||W_l||_2.

    γ_min is the worst-case classification margin
    ``min_i [logit_{y_i}(x_i) - max_{j≠y_i} logit_j(x_i)]``. The spectral
    product is the product of layer-wise operator norms.
    """
    import torch

    t0 = time.perf_counter()
    model.eval()
    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    Y_t = torch.as_tensor(np.asarray(Y).astype(np.int64), device=device)
    with torch.no_grad():
        logits = model(X_t)
    correct = logits.gather(1, Y_t.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, Y_t.view(-1, 1), float("-inf"))
    other_max = masked.max(dim=1).values
    margins = (correct - other_max).cpu().numpy()
    gamma_min = float(margins.min())
    gamma_mean = float(margins.mean())

    spectral_prod = 1.0
    for W in _layer_weights(model):
        spectral_prod *= float(np.linalg.svd(W, compute_uv=False)[0])

    ratio = gamma_min / spectral_prod if spectral_prod > 0 else float("nan")
    return {
        "gamma_min": gamma_min,
        "gamma_mean": gamma_mean,
        "spectral_prod": float(spectral_prod),
        "spectral_margin_ratio": float(ratio),
        "wall": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Sharpness (Hessian top-k via Lanczos)
# ---------------------------------------------------------------------------
def _flatten_grads(grads) -> "torch.Tensor":  # noqa: F821
    import torch

    return torch.cat([g.contiguous().view(-1) for g in grads])


def sharpness(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    n_subsample: int = 1024,
    seed: int = 0,
    device: str = "cpu",
    ncv: int = 20,
    maxiter: int = 300,
    tol: float = 1e-3,
) -> Dict[str, float]:
    """Top-``k`` Hessian eigenvalues of cross-entropy loss at θ via Lanczos.

    The Hessian-vector product is computed by double-backprop on a fixed
    subsample of (X, Y) (deterministic via ``seed``). ``eigsh(which="LA")``
    finds the algebraically largest eigenvalues.
    """
    import torch
    import torch.nn as nn
    from scipy.sparse.linalg import LinearOperator, eigsh

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N > n_subsample:
        idx = rng.choice(N, size=n_subsample, replace=False)
        X_sub = X[idx]
        Y_sub = Y[idx]
    else:
        X_sub = X
        Y_sub = Y
    n_used = X_sub.shape[0]

    X_t = torch.as_tensor(X_sub.astype(np.float32), device=device)
    Y_t = torch.as_tensor(Y_sub.astype(np.int64), device=device)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    criterion = nn.CrossEntropyLoss()

    def hvp(v_np: np.ndarray) -> np.ndarray:
        v = torch.as_tensor(v_np.astype(np.float32), device=device)
        v_split = []
        off = 0
        for p in params:
            n = p.numel()
            v_split.append(v[off : off + n].view_as(p))
            off += n
        loss = criterion(model(X_t), Y_t)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * vp).sum() for g, vp in zip(grads, v_split))
        Hv = torch.autograd.grad(dot, params, retain_graph=False)
        return _flatten_grads(Hv).detach().cpu().numpy().astype(np.float64)

    op = LinearOperator((n_params, n_params), matvec=hvp, dtype=np.float64)
    converged = True
    try:
        eigs = eigsh(
            op, k=k, which="LA",
            ncv=min(ncv, n_params - 1),
            maxiter=maxiter, tol=tol, return_eigenvectors=False,
        )
    except Exception as exc:  # convergence failure → fall back to power iteration
        converged = False
        eigs = _power_iteration_top1(op, n_params, n_iter=200, tol=tol)

    eigs = np.asarray(eigs).ravel()
    return {
        "lambda_max": float(eigs.max()) if eigs.size else float("nan"),
        "lambda_sum_top": float(eigs.sum()) if eigs.size else float("nan"),
        "k": int(k),
        "n_subsample": int(n_used),
        "n_params": int(n_params),
        "converged": bool(converged),
        "wall": time.perf_counter() - t0,
    }


def _power_iteration_top1(
    op, n_params: int, n_iter: int = 200, tol: float = 1e-3
) -> np.ndarray:
    """Single-eigenvalue fallback when Lanczos fails (degenerate Hessian)."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n_params)
    v /= np.linalg.norm(v)
    lam_prev = 0.0
    for _ in range(n_iter):
        Hv = op.matvec(v)
        lam = float(v @ Hv)
        nrm = np.linalg.norm(Hv)
        if nrm < 1e-12:
            return np.array([lam])
        v = Hv / nrm
        if abs(lam - lam_prev) < tol * max(1.0, abs(lam_prev)):
            break
        lam_prev = lam
    return np.array([lam])
