"""
Phase C-M2: extend Recipes 1-4 from `claude_new_estimator_instructions.md`
to ReLU CNNs of the LeNet-5 family.

A "layer" still indexes the ReLU-emitting steps; LeNet-5 (`conv-relu1`,
`conv-relu2`, `fc-relu1`, `fc-relu2`) has L = 4. The cumulative pattern at
depth `l` includes both the ReLU bool masks and the maxpool argmax indices
that precede the ReLU at that depth -- two samples sharing ReLU patterns
but differing in pool argmax do *not* share an active subnetwork.

Active-subnetwork construction reuses the FC recursion
$\\tilde A_i = W^i[S_i, S_{i-1}] \\tilde A_{i-1}$ with three step kinds:

- conv (sparse unrolled $W$, broadcast $b$, gated by next ReLU mask)
- maxpool (sparse 0/1 selector with one nonzero per row at the region's
  argmax position; $b = 0$, no extra gating)
- fc (dense $W$, $b$, gated by ReLU mask)

The unrolled conv weights depend only on the network and are cached per
(epoch, layer); pool selectors are per-region.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from src_experiment.functional_quotient import (
    DEFAULT_EPSILONS,
    QuotientResult,
    _RegionActive,
    cluster_functional,
    routing_information_quotient,
)
from src_experiment.routing_estimator import (
    routing_information,
    truncation_probability,
)
from src_experiment.rtg_analyzer import (
    hamming1_adjacency,
    rtg_diagnostics,
)
from src_experiment.rtg_overlap import (
    region_dominant_class,
    routing_loss_proxy,
)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Spec / weights
# ---------------------------------------------------------------------------
@dataclass
class LeNetSpec:
    """Geometry of a LeNet-5 family network. Derived shapes computed in
    ``__post_init__``."""

    input_shape: Tuple[int, ...]
    conv_channels: Tuple[int, ...]
    fc_widths: Tuple[int, ...]
    kernel_size: int = 5
    pool_size: int = 2
    num_classes: int = 10

    conv_out_shapes: List[Tuple[int, int, int]] = field(default_factory=list)
    pool_out_shapes: List[Tuple[int, int, int]] = field(default_factory=list)
    flat_dim: int = 0

    def __post_init__(self):
        if self.conv_channels:
            c, h, w = self.input_shape
            for out_c in self.conv_channels:
                h_after_conv = h - self.kernel_size + 1
                w_after_conv = w - self.kernel_size + 1
                self.conv_out_shapes.append((out_c, h_after_conv, w_after_conv))
                h_after_pool = h_after_conv // self.pool_size
                w_after_pool = w_after_conv // self.pool_size
                self.pool_out_shapes.append((out_c, h_after_pool, w_after_pool))
                c, h, w = out_c, h_after_pool, w_after_pool
            self.flat_dim = c * h * w
        else:
            self.flat_dim = int(np.prod(self.input_shape))

    @property
    def n_conv(self) -> int:
        return len(self.conv_channels)

    @property
    def n_fc_hidden(self) -> int:
        return len(self.fc_widths)

    @property
    def num_relu_layers(self) -> int:
        return self.n_conv + self.n_fc_hidden

    @property
    def input_size(self) -> int:
        return int(np.prod(self.input_shape))


def parse_lenet_spec_from_h5(h5_path: PathLike) -> LeNetSpec:
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["metadata"].attrs)
    arch_type = str(attrs.get("inferred_arch_type", attrs.get("arch_type", ""))).lower()
    if arch_type != "lenet5":
        raise ValueError(f"HDF5 arch_type is not lenet5 (got {arch_type!r})")

    input_shape = tuple(
        int(x)
        for x in attrs.get("inferred_input_shape", attrs.get("input_shape", [1, 28, 28]))
    )
    conv_channels = tuple(int(x) for x in attrs["conv_channels"])
    fc_widths = tuple(int(x) for x in attrs["fc_widths"])
    kernel_size = int(attrs.get("kernel_size", 5))
    pool_size = int(attrs.get("pool_size", 2))
    num_classes = int(attrs.get("inferred_num_classes", 10))
    return LeNetSpec(
        input_shape=input_shape,
        conv_channels=conv_channels,
        fc_widths=fc_widths,
        kernel_size=kernel_size,
        pool_size=pool_size,
        num_classes=num_classes,
    )


@dataclass
class LeNetWeights:
    """Per-epoch weights. ``fc_W``/``fc_b`` includes the output layer at
    index ``-1``; ReLU is applied to indices ``0 .. n_fc_hidden - 1``."""

    conv_W: List[np.ndarray]
    conv_b: List[np.ndarray]
    fc_W: List[np.ndarray]
    fc_b: List[np.ndarray]


def load_lenet_weights(h5_path: PathLike, epoch: int) -> LeNetWeights:
    with h5py.File(h5_path, "r") as f:
        grp = f[f"epochs/epoch_{epoch}"]
        conv_W: List[np.ndarray] = []
        conv_b: List[np.ndarray] = []
        i = 1
        while f"conv{i}.weight" in grp:
            conv_W.append(np.asarray(grp[f"conv{i}.weight"][:], dtype=np.float32))
            conv_b.append(np.asarray(grp[f"conv{i}.bias"][:], dtype=np.float32))
            i += 1
        fc_W: List[np.ndarray] = []
        fc_b: List[np.ndarray] = []
        j = 1
        while f"fc{j}.weight" in grp:
            fc_W.append(np.asarray(grp[f"fc{j}.weight"][:], dtype=np.float32))
            fc_b.append(np.asarray(grp[f"fc{j}.bias"][:], dtype=np.float32))
            j += 1
    return LeNetWeights(conv_W=conv_W, conv_b=conv_b, fc_W=fc_W, fc_b=fc_b)


# ---------------------------------------------------------------------------
# Forward pass: per-step gates
# ---------------------------------------------------------------------------
@dataclass
class CNNPatterns:
    """Per-sample gating patterns for the LeNet5 forward pass.

    Conv-ReLU bool masks and FC-ReLU bool masks share the bit semantics of
    the FC ``forward_activation_patterns`` (each bit = ``z > 0``). Pool
    argmax indices are per-region channel-major flat indices into the
    preceding conv-ReLU step's output space.
    """

    conv_relu_masks: List[np.ndarray]
    pool_argmax_indices: List[np.ndarray]
    fc_relu_masks: List[np.ndarray]


def forward_activation_patterns_cnn(
    spec: LeNetSpec,
    weights: LeNetWeights,
    X: np.ndarray,
) -> CNNPatterns:
    """Run a LeNet-5 forward pass and emit per-step gating patterns.

    ``X`` may be ``(N, *input_shape)`` or flattened ``(N, prod(input_shape))``;
    HDF5 stores ``points`` flattened, so the latter is the common case.
    """
    if len(weights.conv_W) != spec.n_conv:
        raise ValueError(
            f"weights.conv_W has {len(weights.conv_W)} layers, expected {spec.n_conv}"
        )
    if len(weights.fc_W) != spec.n_fc_hidden + 1:
        raise ValueError(
            f"weights.fc_W has {len(weights.fc_W)} layers, expected "
            f"{spec.n_fc_hidden + 1} (hidden + output)"
        )

    N = X.shape[0]
    x = torch.as_tensor(X, dtype=torch.float32)
    if x.dim() == 2 and spec.n_conv > 0:
        x = x.view(N, *spec.input_shape)

    conv_relu_masks: List[np.ndarray] = []
    pool_argmax_indices: List[np.ndarray] = []

    for i in range(spec.n_conv):
        W = torch.from_numpy(weights.conv_W[i])
        b = torch.from_numpy(weights.conv_b[i])
        z = F.conv2d(x, W, b, stride=1, padding=0)
        mask = z > 0
        conv_relu_masks.append(mask.reshape(N, -1).cpu().numpy().astype(bool))

        z_post_relu = torch.where(mask, z, torch.zeros_like(z))
        pooled, idx_per_channel = F.max_pool2d(
            z_post_relu, kernel_size=spec.pool_size, return_indices=True
        )
        # idx_per_channel: (N, C, H', W') indices into per-channel flat (H_in*W_in).
        # Convert to channel-major flat indices into (C * H_in * W_in).
        C, H_in, W_in = spec.conv_out_shapes[i]
        c_offsets = (
            torch.arange(C, dtype=idx_per_channel.dtype) * H_in * W_in
        ).view(1, C, 1, 1)
        idx_flat = idx_per_channel + c_offsets
        pool_argmax_indices.append(
            idx_flat.reshape(N, -1).cpu().numpy().astype(np.int64)
        )

        x = pooled

    if spec.n_conv > 0:
        a = x.reshape(N, -1)
    else:
        a = x if x.dim() == 2 else x.reshape(N, -1)

    fc_relu_masks: List[np.ndarray] = []
    for j in range(spec.n_fc_hidden):
        W = torch.from_numpy(weights.fc_W[j])
        bvec = torch.from_numpy(weights.fc_b[j])
        z = a @ W.T + bvec
        mask = z > 0
        fc_relu_masks.append(mask.cpu().numpy().astype(bool))
        a = torch.where(mask, z, torch.zeros_like(z))

    return CNNPatterns(
        conv_relu_masks=conv_relu_masks,
        pool_argmax_indices=pool_argmax_indices,
        fc_relu_masks=fc_relu_masks,
    )


# ---------------------------------------------------------------------------
# Cumulative-pattern hashing
# ---------------------------------------------------------------------------
def _cumulative_bytes_at_layer(
    patterns: CNNPatterns,
    spec: LeNetSpec,
    layer: int,
) -> np.ndarray:
    """Return ``(N, n_bytes)`` uint8 cumulative bytes used for hashing.

    Concatenates packed ReLU bools and raw int64-bytes from pool argmaxes in
    the order in which they appear in the forward pass up to ``layer``.
    """
    if layer < 1 or layer > spec.num_relu_layers:
        raise ValueError(f"layer must be 1..{spec.num_relu_layers}")

    N = (
        patterns.conv_relu_masks[0].shape[0]
        if patterns.conv_relu_masks
        else patterns.fc_relu_masks[0].shape[0]
    )
    parts: List[np.ndarray] = []

    n_conv_used = min(layer, spec.n_conv)
    for i in range(n_conv_used):
        parts.append(np.packbits(patterns.conv_relu_masks[i], axis=1))
        # Pool i sits between conv-relu i and conv-relu i+1; include only if
        # we're crossing past it (i.e. there's another step ahead in the
        # cumulative pattern).
        if i < n_conv_used - 1 or layer > spec.n_conv:
            parts.append(
                patterns.pool_argmax_indices[i].view(np.uint8).reshape(N, -1)
            )

    n_fc_used = max(layer - spec.n_conv, 0)
    for j in range(n_fc_used):
        parts.append(np.packbits(patterns.fc_relu_masks[j], axis=1))

    return np.concatenate(parts, axis=1)


def cumulative_pattern_hashes_cnn(
    patterns: CNNPatterns,
    spec: LeNetSpec,
    layer: int,
) -> np.ndarray:
    """md5 hashes of per-sample cumulative patterns up to depth ``layer``."""
    cumulative = _cumulative_bytes_at_layer(patterns, spec, layer)
    N = cumulative.shape[0]
    digests = np.empty(N, dtype=object)
    for i in range(N):
        digests[i] = hashlib.md5(cumulative[i].tobytes()).digest()
    return digests


# ---------------------------------------------------------------------------
# Conv unrolling (cached per epoch)
# ---------------------------------------------------------------------------
def unrolled_conv_W_b(
    spec: LeNetSpec,
    weights: LeNetWeights,
    i: int,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return the sparse unrolled conv matrix and broadcast bias for layer ``i``.

    Output index = ``c_o * H_out * W_out + h * W_out + w``;
    input index  = ``c_i * H_in  * W_in  + (h+ki) * W_in + (w+kj)``.
    """
    W_conv = weights.conv_W[i]
    b_conv = weights.conv_b[i]
    out_c, in_c, k, _ = W_conv.shape

    in_shape = spec.input_shape if i == 0 else spec.pool_out_shapes[i - 1]
    in_C, H_in, W_in = in_shape
    if in_C != in_c:
        raise ValueError(
            f"conv {i}: weight in_c={in_c} != input C={in_C}"
        )
    out_C, H_out, W_out = spec.conv_out_shapes[i]
    n_out = out_C * H_out * W_out
    n_in = in_C * H_in * W_in

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    data: List[np.ndarray] = []
    h_arr = np.arange(H_out).reshape(-1, 1, 1, 1)
    w_arr = np.arange(W_out).reshape(1, -1, 1, 1)
    ki_arr = np.arange(k).reshape(1, 1, -1, 1)
    kj_arr = np.arange(k).reshape(1, 1, 1, -1)
    for c_o in range(out_C):
        out_idx = (
            c_o * H_out * W_out + h_arr * W_out + w_arr
        )  # (H_out, W_out, 1, 1)
        for c_i in range(in_c):
            in_idx = (
                c_i * H_in * W_in
                + (h_arr + ki_arr) * W_in
                + (w_arr + kj_arr)
            )  # (H_out, W_out, k, k)
            vals = np.broadcast_to(
                W_conv[c_o, c_i][None, None, :, :],
                (H_out, W_out, k, k),
            )
            rows.append(np.broadcast_to(out_idx, (H_out, W_out, k, k)).ravel())
            cols.append(in_idx.ravel())
            data.append(vals.ravel())

    rows_a = np.concatenate(rows)
    cols_a = np.concatenate(cols)
    data_a = np.concatenate(data).astype(np.float32, copy=False)
    W_sparse = sp.csr_matrix((data_a, (rows_a, cols_a)), shape=(n_out, n_in))
    b_unroll = np.repeat(b_conv, H_out * W_out).astype(np.float32)
    return W_sparse, b_unroll


# ---------------------------------------------------------------------------
# Active subnetwork (per region)
# ---------------------------------------------------------------------------
@dataclass
class _RegionGates:
    """Per-region gating extracted from `CNNPatterns` for a single sample idx."""

    conv_relu: List[np.ndarray]   # bool, flat (C*H*W)
    pool_argmax: List[np.ndarray]  # int64, flat (C*H'*W')
    fc_relu: List[np.ndarray]     # bool, flat (width,)


def _gates_from_patterns_for_sample(
    patterns: CNNPatterns,
    sample_idx: int,
) -> _RegionGates:
    return _RegionGates(
        conv_relu=[m[sample_idx] for m in patterns.conv_relu_masks],
        pool_argmax=[a[sample_idx] for a in patterns.pool_argmax_indices],
        fc_relu=[m[sample_idx] for m in patterns.fc_relu_masks],
    )


def compute_active_subnetwork_cnn(
    spec: LeNetSpec,
    weights: LeNetWeights,
    gates: _RegionGates,
    layer: int,
    unrolled_convs: Optional[Sequence[Tuple[sp.csr_matrix, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Active subnetwork (Ã, c̃, S_l) at hidden ReLU layer ``layer`` for one region.

    ``unrolled_convs`` may be passed in to amortize the (expensive) unroll
    across many regions sharing the same weights.
    """
    if layer < 1 or layer > spec.num_relu_layers:
        raise ValueError(f"layer must be 1..{spec.num_relu_layers}")

    n_0 = spec.input_size
    tilde_A = np.eye(n_0, dtype=np.float32)
    tilde_c = np.zeros(n_0, dtype=np.float32)
    S_prev = np.arange(n_0)

    if unrolled_convs is None:
        unrolled_convs = [
            unrolled_conv_W_b(spec, weights, i) for i in range(spec.n_conv)
        ]

    for i in range(min(layer, spec.n_conv)):
        W_full, b_full = unrolled_convs[i]
        mask = gates.conv_relu[i]
        S_curr = np.where(mask)[0]
        # Sparse row-then-column slice; convert to dense for the matmul.
        W_step = W_full[S_curr, :].toarray()[:, S_prev]
        b_step = b_full[S_curr]
        tilde_A = W_step @ tilde_A
        tilde_c = W_step @ tilde_c + b_step
        S_prev = S_curr

        if i + 1 == layer:
            return tilde_A, tilde_c, S_prev

        # Maxpool: per-row 0/1 selector at the region's argmax position.
        argmax = gates.pool_argmax[i]
        C, H_in, W_in = spec.conv_out_shapes[i]
        Cp, Hp, Wp = spec.pool_out_shapes[i]
        n_pool_in = C * H_in * W_in
        n_pool_out = Cp * Hp * Wp

        prev_idx_to_pos = -np.ones(n_pool_in, dtype=np.int64)
        prev_idx_to_pos[S_prev] = np.arange(len(S_prev))
        col_pos = prev_idx_to_pos[argmax]  # (-1) where argmax falls in dead row
        valid_rows = np.where(col_pos >= 0)[0]

        W_pool = np.zeros((n_pool_out, len(S_prev)), dtype=np.float32)
        if valid_rows.size:
            W_pool[valid_rows, col_pos[valid_rows]] = 1.0

        tilde_A = W_pool @ tilde_A
        tilde_c = W_pool @ tilde_c
        S_prev = np.arange(n_pool_out)

    fc_used = max(layer - spec.n_conv, 0)
    for j in range(fc_used):
        W = weights.fc_W[j]
        b = weights.fc_b[j]
        mask = gates.fc_relu[j]
        S_curr = np.where(mask)[0]
        W_step = W[np.ix_(S_curr, S_prev)]
        b_step = b[S_curr]
        tilde_A = W_step @ tilde_A
        tilde_c = W_step @ tilde_c + b_step
        S_prev = S_curr

    return tilde_A, tilde_c, S_prev


# ---------------------------------------------------------------------------
# Single-walk chain advancement
# ---------------------------------------------------------------------------
@dataclass
class _ChainState:
    """Persistent walk state for a single region. ``layer`` tracks the index
    of the most recently completed ReLU snapshot (0 = before conv-relu1)."""

    tilde_A: np.ndarray
    tilde_c: np.ndarray
    S_prev: np.ndarray
    layer: int


def _init_chain_state(spec: LeNetSpec) -> _ChainState:
    n_0 = spec.input_size
    return _ChainState(
        tilde_A=np.eye(n_0, dtype=np.float32),
        tilde_c=np.zeros(n_0, dtype=np.float32),
        S_prev=np.arange(n_0),
        layer=0,
    )


def _apply_conv_relu(
    state: _ChainState,
    spec: LeNetSpec,
    gates: _RegionGates,
    conv_idx: int,
    unrolled_convs: Sequence[Tuple[sp.csr_matrix, np.ndarray]],
) -> None:
    W_full, b_full = unrolled_convs[conv_idx]
    mask = gates.conv_relu[conv_idx]
    S_curr = np.where(mask)[0]
    W_step = W_full[S_curr, :].toarray()[:, state.S_prev]
    b_step = b_full[S_curr]
    state.tilde_A = W_step @ state.tilde_A
    state.tilde_c = W_step @ state.tilde_c + b_step
    state.S_prev = S_curr


def _apply_pool(
    state: _ChainState,
    spec: LeNetSpec,
    gates: _RegionGates,
    pool_idx: int,
) -> None:
    argmax = gates.pool_argmax[pool_idx]
    C, H_in, W_in = spec.conv_out_shapes[pool_idx]
    Cp, Hp, Wp = spec.pool_out_shapes[pool_idx]
    n_pool_in = C * H_in * W_in
    n_pool_out = Cp * Hp * Wp

    prev_idx_to_pos = -np.ones(n_pool_in, dtype=np.int64)
    prev_idx_to_pos[state.S_prev] = np.arange(len(state.S_prev))
    col_pos = prev_idx_to_pos[argmax]
    valid_rows = np.where(col_pos >= 0)[0]

    W_pool = np.zeros((n_pool_out, len(state.S_prev)), dtype=np.float32)
    if valid_rows.size:
        W_pool[valid_rows, col_pos[valid_rows]] = 1.0
    state.tilde_A = W_pool @ state.tilde_A
    state.tilde_c = W_pool @ state.tilde_c
    state.S_prev = np.arange(n_pool_out)


def _apply_fc_relu(
    state: _ChainState,
    weights: LeNetWeights,
    gates: _RegionGates,
    fc_idx: int,
) -> None:
    W = weights.fc_W[fc_idx]
    b = weights.fc_b[fc_idx]
    mask = gates.fc_relu[fc_idx]
    S_curr = np.where(mask)[0]
    W_step = W[np.ix_(S_curr, state.S_prev)]
    b_step = b[S_curr]
    state.tilde_A = W_step @ state.tilde_A
    state.tilde_c = W_step @ state.tilde_c + b_step
    state.S_prev = S_curr


def _advance_chain_state(
    state: _ChainState,
    spec: LeNetSpec,
    weights: LeNetWeights,
    gates: _RegionGates,
    target_layer: int,
    unrolled_convs: Sequence[Tuple[sp.csr_matrix, np.ndarray]],
) -> None:
    """Mutate ``state`` forward to the snapshot point of ``target_layer``.

    Caller must request layers in non-decreasing order; the state caches
    the previous snapshot so each step does only the incremental work to
    reach the next layer's ReLU boundary.
    """
    if target_layer < state.layer:
        raise ValueError(
            f"target_layer={target_layer} < state.layer={state.layer}; "
            "advance is forward-only"
        )
    if target_layer > spec.num_relu_layers:
        raise ValueError(
            f"target_layer={target_layer} exceeds L={spec.num_relu_layers}"
        )

    while state.layer < target_layer:
        next_layer = state.layer + 1
        if next_layer <= spec.n_conv:
            # Crossing into conv-relu(next_layer). Apply prior pool first.
            if state.layer >= 1:
                _apply_pool(state, spec, gates, state.layer - 1)
            _apply_conv_relu(state, spec, gates, next_layer - 1, unrolled_convs)
        else:
            fc_idx = next_layer - spec.n_conv - 1
            if state.layer == spec.n_conv and spec.n_conv > 0:
                # First fc-relu after the last conv-relu: cross the final pool.
                _apply_pool(state, spec, gates, spec.n_conv - 1)
            _apply_fc_relu(state, weights, gates, fc_idx)
        state.layer = next_layer


# ---------------------------------------------------------------------------
# Per-region representative gates + RTG cumulative bool reconstruction
# ---------------------------------------------------------------------------
def collect_unique_region_gates(
    patterns: CNNPatterns,
    omega_ids: np.ndarray,
) -> Dict[bytes, _RegionGates]:
    """Pick the first sample per region and snapshot its full gate stack."""
    first_idx: Dict[bytes, int] = {}
    for i, w in enumerate(omega_ids):
        if w not in first_idx:
            first_idx[w] = i
    return {
        rid: _gates_from_patterns_for_sample(patterns, idx)
        for rid, idx in first_idx.items()
    }


def cumulative_relu_bits_per_region(
    patterns: CNNPatterns,
    spec: LeNetSpec,
    omega_ids: np.ndarray,
    layer: int,
) -> Dict[bytes, np.ndarray]:
    """Map each unique region to its cumulative *ReLU* bool pattern up to ``layer``.

    Drops the pool-argmax bytes -- the RTG (Recipe 4) is defined on
    Hamming-1 ReLU flips per the paper. Pool argmax lives in a different
    code space and isn't a single-bit object; including it would wreck the
    bit-flip semantics.
    """
    first_idx: Dict[bytes, int] = {}
    for i, w in enumerate(omega_ids):
        if w not in first_idx:
            first_idx[w] = i

    parts: List[np.ndarray] = []
    n_conv_used = min(layer, spec.n_conv)
    for i in range(n_conv_used):
        parts.append(patterns.conv_relu_masks[i].astype(bool, copy=False))
    n_fc_used = max(layer - spec.n_conv, 0)
    for j in range(n_fc_used):
        parts.append(patterns.fc_relu_masks[j].astype(bool, copy=False))
    cumulative = np.concatenate(parts, axis=1)

    return {rid: cumulative[idx].copy() for rid, idx in first_idx.items()}


# ---------------------------------------------------------------------------
# Stratified subsample
# ---------------------------------------------------------------------------
def _stratified_subsample_indices(
    labels: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``n`` indices roughly proportional to class frequency.

    Per-class counts are rounded; the largest class absorbs any rounding
    residual so the total is exactly ``n``. If a class has fewer samples
    than its share, every member is taken and the residual is redistributed
    by re-rounding.
    """
    classes, counts = np.unique(labels, return_counts=True)
    fractions = counts / counts.sum()
    per_class = np.floor(fractions * n).astype(np.int64)
    deficit = n - per_class.sum()
    # distribute the rounding deficit to classes with largest fractional remainder
    remainders = fractions * n - per_class
    if deficit > 0:
        order = np.argsort(-remainders)
        per_class[order[:deficit]] += 1

    chunks: List[np.ndarray] = []
    leftover = 0
    for cls, k in zip(classes, per_class):
        cls_idx = np.where(labels == cls)[0]
        take = min(k, len(cls_idx))
        leftover += k - take
        if take > 0:
            chunks.append(rng.choice(cls_idx, size=take, replace=False))
    # Fill any leftover from a global pool (rare; only triggers if a class
    # was undersized vs its share).
    if leftover > 0:
        all_idx = np.arange(len(labels))
        already = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        pool = np.setdiff1d(all_idx, already, assume_unique=False)
        chunks.append(rng.choice(pool, size=leftover, replace=False))
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class FunctionalQuotientEstimatorCNN:
    """LeNet5 analogue of :class:`FunctionalQuotientEstimator`.

    Mirrors the FC driver's columns. Hidden-layer index runs 1..L with
    L = n_conv + n_fc_hidden. Pool argmax enters the cumulative pattern
    used for region IDs and active-subnetwork construction; the RTG
    (Recipe 4) is built on the Hamming-1 ReLU bits only.
    """

    def __init__(
        self,
        h5_path: PathLike,
        probe_subsample: Optional[int] = None,
        probe_seed: int = 0,
        stratify: bool = True,
    ):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        self.spec = parse_lenet_spec_from_h5(self.h5_path)

        with h5py.File(self.h5_path, "r") as f:
            attrs = dict(f["metadata"].attrs)
            self.network_id = str(
                attrs.get("experiment_name", self.h5_path.stem)
            )
            self.seed = int(attrs.get("model_seed", -1))
            self.epochs = sorted(
                int(k.split("_")[1])
                for k in f["epochs"].keys()
                if k.startswith("epoch_")
            )
            full_points = np.asarray(f["points"][:], dtype=np.float32)
            full_labels = np.asarray(f["labels"][:], dtype=np.int64)

        self.probe_full_size = full_points.shape[0]
        if probe_subsample is not None and probe_subsample < self.probe_full_size:
            rng = np.random.default_rng(probe_seed)
            if stratify:
                idx = _stratified_subsample_indices(
                    full_labels, probe_subsample, rng
                )
            else:
                idx = rng.choice(
                    self.probe_full_size, size=probe_subsample, replace=False
                )
            idx.sort()
            self.subsample_indices: Optional[np.ndarray] = idx
            self.points = full_points[idx]
            self.labels = full_labels[idx]
        else:
            self.subsample_indices = None
            self.points = full_points
            self.labels = full_labels

    # convenience wrapper to keep symmetry with the FC class
    @property
    def num_hidden_layers(self) -> int:
        return self.spec.num_relu_layers

    def evaluate_epoch(
        self,
        epoch: int,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        X_holdout: Optional[np.ndarray] = None,
        y_holdout: Optional[np.ndarray] = None,
        epsilons: Sequence[float] = DEFAULT_EPSILONS,
    ) -> List[QuotientResult]:
        if X is None:
            X, y = self.points, self.labels
        if y is None:
            raise ValueError("y must be provided when X is provided")

        weights = load_lenet_weights(self.h5_path, epoch)
        patterns = forward_activation_patterns_cnn(self.spec, weights, X)

        holdout_patterns = None
        if X_holdout is not None:
            if y_holdout is None:
                raise ValueError("y_holdout must accompany X_holdout")
            holdout_patterns = forward_activation_patterns_cnn(
                self.spec, weights, X_holdout
            )

        N = len(y)
        num_classes = int(np.max(y)) + 1
        unrolled_convs = [
            unrolled_conv_W_b(self.spec, weights, i)
            for i in range(self.spec.n_conv)
        ]

        # Region IDs at the deepest layer subsume those at every shallower
        # layer (cumulative pattern is a prefix). Build the per-region gate
        # snapshot once, keyed by deepest-layer rid, and reuse it across
        # layers via persistent _ChainState advancement.
        omega_deepest = cumulative_pattern_hashes_cnn(
            patterns, self.spec, self.spec.num_relu_layers
        )
        region_gates_deepest = collect_unique_region_gates(patterns, omega_deepest)
        # Lazy-init: 784x784 eye buffers cost ~2.5MB each; only allocate for
        # deep reps that actually get selected.
        chain_states: Dict[bytes, _ChainState] = {}

        out: List[QuotientResult] = []
        for layer in range(1, self.spec.num_relu_layers + 1):
            omega = cumulative_pattern_hashes_cnn(patterns, self.spec, layer)
            plug_in, mm, R, H_Y = routing_information(
                omega, y, num_classes=num_classes
            )
            rho = R / N

            tp = float("nan")
            if holdout_patterns is not None:
                omega_h = cumulative_pattern_hashes_cnn(
                    holdout_patterns, self.spec, layer
                )
                tp = truncation_probability(omega, omega_h)

            # Advance every region's chain to ``layer`` (mutates in place).
            # Map each shallow-layer rid to the active subnetwork by picking
            # any deepest-layer rid that resolves to it; ε-clustering then
            # operates on the shallow rids' active data.
            shallow_to_deep_rep: Dict[bytes, bytes] = {}
            for i, w_shallow in enumerate(omega):
                if w_shallow not in shallow_to_deep_rep:
                    shallow_to_deep_rep[w_shallow] = omega_deepest[i]

            active_data: Dict[bytes, _RegionActive] = {}
            for rid_shallow, rid_deep in shallow_to_deep_rep.items():
                if rid_deep not in chain_states:
                    chain_states[rid_deep] = _init_chain_state(self.spec)
                state = chain_states[rid_deep]
                _advance_chain_state(
                    state,
                    self.spec,
                    weights,
                    region_gates_deepest[rid_deep],
                    layer,
                    unrolled_convs,
                )
                active_data[rid_shallow] = _RegionActive(
                    state.tilde_A, state.tilde_c, state.S_prev
                )

            relu_patterns = cumulative_relu_bits_per_region(
                patterns, self.spec, omega, layer
            )
            adjacency = hamming1_adjacency(relu_patterns)
            rtg = rtg_diagnostics(adjacency)
            dominant = region_dominant_class(omega, y)
            rl = routing_loss_proxy(adjacency, dominant)

            for eps in epsilons:
                quotient_map, num_q = cluster_functional(active_data, eps)
                pi_func, mm_func, _, _ = routing_information_quotient(
                    omega, y, quotient_map, num_classes=num_classes
                )
                out.append(
                    QuotientResult(
                        layer=layer,
                        epsilon=float(eps),
                        N=N,
                        num_regions=R,
                        num_quotient=num_q,
                        rho=rho,
                        rho_func=num_q / R if R > 0 else float("nan"),
                        plug_in_bits=plug_in,
                        miller_madow_bits=mm,
                        plug_in_func_bits=pi_func,
                        miller_madow_func_bits=mm_func,
                        H_Y_bits=H_Y,
                        truncation_prob=tp,
                        num_rtg_components=rtg.num_components,
                        rtg_largest_component_frac=rtg.largest_component_frac,
                        rtg_isolated_frac=rtg.isolated_frac,
                        rl_proxy=rl,
                    )
                )
        return out

    def evaluate_all(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        X_holdout: Optional[np.ndarray] = None,
        y_holdout: Optional[np.ndarray] = None,
        epsilons: Sequence[float] = DEFAULT_EPSILONS,
    ) -> pd.DataFrame:
        rows = []
        for ep in self.epochs:
            for r in self.evaluate_epoch(
                ep,
                X=X,
                y=y,
                X_holdout=X_holdout,
                y_holdout=y_holdout,
                epsilons=epsilons,
            ):
                rows.append(
                    {
                        "network_id": self.network_id,
                        "epoch": ep,
                        "layer": r.layer,
                        "epsilon": r.epsilon,
                        "seed": self.seed,
                        "N": r.N,
                        "num_regions": r.num_regions,
                        "num_quotient": r.num_quotient,
                        "rho": r.rho,
                        "rho_func": r.rho_func,
                        "plug_in_bits": r.plug_in_bits,
                        "miller_madow_bits": r.miller_madow_bits,
                        "plug_in_func_bits": r.plug_in_func_bits,
                        "miller_madow_func_bits": r.miller_madow_func_bits,
                        "H_Y_bits": r.H_Y_bits,
                        "truncation_prob": r.truncation_prob,
                        "num_rtg_components": r.num_rtg_components,
                        "rtg_largest_component_frac": r.rtg_largest_component_frac,
                        "rtg_isolated_frac": r.rtg_isolated_frac,
                        "rl_proxy": r.rl_proxy,
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "usage: python -m src_experiment.cnn_estimator <path/to/file.h5>"
        )
        sys.exit(1)
    estimator = FunctionalQuotientEstimatorCNN(sys.argv[1])
    df = estimator.evaluate_all()
    print(df.to_string(index=False))
