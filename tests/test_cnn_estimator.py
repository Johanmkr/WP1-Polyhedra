"""Phase C-M2 smoke tests for `src_experiment.cnn_estimator`.

Two correctness gates:

1. **Activation equivalence** (primary) -- on a small synthetic LeNet,
   the per-region active subnetwork (Ã, c̃) must reproduce the actual
   post-ReLU activation for any sample in that region: Ã x + c̃ equals
   the live entries of the network's hidden activation at depth ``l``.

2. **FC reduction** (sanity) -- with ``conv_channels=()`` the CNN driver
   must produce the same ``omega_ids`` and the same ``(Ã, c̃)`` per
   region as the existing FC ``FunctionalQuotientEstimator``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src_experiment.cnn_estimator import (  # noqa: E402
    LeNetSpec,
    LeNetWeights,
    _advance_chain_state,
    _gates_from_patterns_for_sample,
    _init_chain_state,
    collect_unique_region_gates,
    compute_active_subnetwork_cnn,
    cumulative_pattern_hashes_cnn,
    forward_activation_patterns_cnn,
    unrolled_conv_W_b,
)
from src_experiment.functional_quotient import (  # noqa: E402
    _build_active_data,
    collect_unique_region_patterns,
    compute_active_subnetwork,
)
from src_experiment.routing_estimator import (  # noqa: E402
    cumulative_pattern_hashes,
    forward_activation_patterns,
)


def _ground_truth_hidden_activation_lenet(
    spec: LeNetSpec,
    weights: LeNetWeights,
    x_one: np.ndarray,
    layer: int,
) -> np.ndarray:
    """Run a single sample through LeNet manually and return the live
    entries of the post-ReLU activation at hidden ReLU layer ``layer``."""
    x = torch.as_tensor(x_one, dtype=torch.float32).reshape(1, *spec.input_shape)
    for i in range(spec.n_conv):
        W = torch.from_numpy(weights.conv_W[i])
        b = torch.from_numpy(weights.conv_b[i])
        z = F.conv2d(x, W, b, stride=1, padding=0)
        if i + 1 == layer:
            mask = (z > 0).reshape(-1).numpy()
            return z.reshape(-1).numpy()[mask]
        x = F.max_pool2d(F.relu(z), kernel_size=spec.pool_size)

    a = x.reshape(1, -1)
    for j in range(spec.n_fc_hidden):
        W = torch.from_numpy(weights.fc_W[j])
        bvec = torch.from_numpy(weights.fc_b[j])
        z = a @ W.T + bvec
        if (spec.n_conv + j + 1) == layer:
            mask = (z.reshape(-1) > 0).numpy()
            return z.reshape(-1).numpy()[mask]
        a = F.relu(z)
    raise ValueError(f"layer {layer} out of range")


def make_random_lenet_weights(spec: LeNetSpec, rng: np.random.Generator) -> LeNetWeights:
    conv_W = []
    conv_b = []
    in_c = spec.input_shape[0]
    for out_c in spec.conv_channels:
        conv_W.append(
            rng.standard_normal((out_c, in_c, spec.kernel_size, spec.kernel_size)).astype(
                np.float32
            )
            * 0.5
        )
        conv_b.append(rng.standard_normal(out_c).astype(np.float32) * 0.1)
        in_c = out_c

    fc_W = []
    fc_b = []
    prev = spec.flat_dim
    for w in spec.fc_widths:
        fc_W.append(rng.standard_normal((w, prev)).astype(np.float32) * 0.5)
        fc_b.append(rng.standard_normal(w).astype(np.float32) * 0.1)
        prev = w
    fc_W.append(rng.standard_normal((spec.num_classes, prev)).astype(np.float32) * 0.5)
    fc_b.append(rng.standard_normal(spec.num_classes).astype(np.float32) * 0.1)
    return LeNetWeights(conv_W=conv_W, conv_b=conv_b, fc_W=fc_W, fc_b=fc_b)


def test_activation_equivalence() -> None:
    """Primary gate: Ã x + c̃ matches the live hidden activations for a sample."""
    rng = np.random.default_rng(0)
    spec = LeNetSpec(
        input_shape=(1, 14, 14),
        conv_channels=(3, 4),
        fc_widths=(6, 5),
        kernel_size=3,
        pool_size=2,
        num_classes=4,
    )
    weights = make_random_lenet_weights(spec, rng)
    N = 32
    X = rng.standard_normal((N, spec.input_size)).astype(np.float32)
    patterns = forward_activation_patterns_cnn(spec, weights, X)

    unrolled = [unrolled_conv_W_b(spec, weights, i) for i in range(spec.n_conv)]
    failures = []
    for layer in range(1, spec.num_relu_layers + 1):
        omega = cumulative_pattern_hashes_cnn(patterns, spec, layer)
        rep_for_rid = collect_unique_region_gates(patterns, omega)
        # also need a sample index per rid
        first_idx = {}
        for i, w in enumerate(omega):
            if w not in first_idx:
                first_idx[w] = i
        for rid, gates in rep_for_rid.items():
            tA, tc, S_l = compute_active_subnetwork_cnn(
                spec, weights, gates, layer, unrolled_convs=unrolled
            )
            sample = X[first_idx[rid]]
            predicted = tA @ sample + tc
            ground = _ground_truth_hidden_activation_lenet(
                spec, weights, sample, layer
            )
            assert tA.shape[0] == ground.shape[0], (
                f"layer {layer}: |S_l|={tA.shape[0]} != ground {ground.shape[0]}"
            )
            if ground.size == 0:
                continue
            err = float(np.max(np.abs(predicted - ground)))
            if err > 1e-3:
                failures.append((layer, rid, err))
    assert not failures, f"activation-equivalence failed: {failures[:3]} (first 3)"
    print("[OK] test_activation_equivalence")


def test_fc_reduction() -> None:
    """With conv_channels=(), the CNN driver matches the FC pipeline."""
    rng = np.random.default_rng(1)
    n_in = 12
    spec = LeNetSpec(
        input_shape=(n_in,),  # type: ignore[arg-type]
        conv_channels=(),
        fc_widths=(8, 6),
        kernel_size=1,
        pool_size=1,
        num_classes=3,
    )
    weights = make_random_lenet_weights(spec, rng)
    N = 50
    X = rng.standard_normal((N, n_in)).astype(np.float32)

    # CNN side
    patterns = forward_activation_patterns_cnn(spec, weights, X)

    # FC side: hidden weights only (drop the output layer)
    fc_W_hidden = weights.fc_W[:-1]
    fc_b_hidden = weights.fc_b[:-1]
    fc_patterns = forward_activation_patterns(fc_W_hidden, fc_b_hidden, X)

    # Per-layer ReLU masks must agree exactly.
    assert len(patterns.fc_relu_masks) == len(fc_patterns)
    for j, (cm, fm) in enumerate(zip(patterns.fc_relu_masks, fc_patterns)):
        assert np.array_equal(cm, fm), f"fc-relu mask mismatch at layer {j}"

    # Region IDs must agree at every layer.
    for layer in range(1, spec.num_relu_layers + 1):
        cnn_omega = cumulative_pattern_hashes_cnn(patterns, spec, layer)
        fc_omega = cumulative_pattern_hashes(fc_patterns, layer)
        assert np.array_equal(cnn_omega, fc_omega), (
            f"omega mismatch at layer {layer}"
        )

    # (Ã, c̃) must agree (numerically) for each region at the deepest layer.
    layer = spec.num_relu_layers
    cnn_omega = cumulative_pattern_hashes_cnn(patterns, spec, layer)
    cnn_gates = collect_unique_region_gates(patterns, cnn_omega)
    cnn_active = {}
    for rid, gates in cnn_gates.items():
        tA, tc, S_l = compute_active_subnetwork_cnn(spec, weights, gates, layer)
        cnn_active[rid] = (tA, tc, S_l)

    fc_region_patterns = collect_unique_region_patterns(fc_patterns, cnn_omega, layer)
    fc_active = _build_active_data(fc_W_hidden, fc_b_hidden, fc_region_patterns, layer)

    for rid, (tA, tc, S_l) in cnn_active.items():
        fc = fc_active[rid]
        assert np.array_equal(S_l, fc.S_l), f"S_l mismatch for region {rid.hex()[:8]}"
        assert np.allclose(tA, fc.tilde_A, atol=1e-5), "tilde_A mismatch"
        assert np.allclose(tc, fc.tilde_c, atol=1e-5), "tilde_c mismatch"
    print("[OK] test_fc_reduction")


def test_pool_argmax_in_region_id() -> None:
    """Direct construction: two samples sharing every ReLU mask but differing
    only in pool argmax must collide at layer 1 (pool not yet crossed) and
    diverge at any later layer that crosses past pool 1."""
    from src_experiment.cnn_estimator import CNNPatterns

    spec = LeNetSpec(
        input_shape=(1, 6, 6),
        conv_channels=(2,),
        fc_widths=(3,),
        kernel_size=3,
        pool_size=2,
        num_classes=2,
    )
    # 2 conv-relu outputs of shape (2, 4, 4) = 32 bits; identical for both samples
    relu1 = np.ones((2, 32), dtype=bool)
    # 2 pool argmax arrays of shape (2, 2, 2) = 4 ints; differ in one slot
    argmax = np.zeros((2, 4), dtype=np.int64)
    argmax[1, 0] = 1  # second sample picks a different argmax in cell 0
    fc1 = np.ones((2, 3), dtype=bool)
    patterns = CNNPatterns(
        conv_relu_masks=[relu1],
        pool_argmax_indices=[argmax],
        fc_relu_masks=[fc1],
    )

    omega1 = cumulative_pattern_hashes_cnn(patterns, spec, 1)
    assert omega1[0] == omega1[1], "layer 1 (pre-pool) should collide"

    omega2 = cumulative_pattern_hashes_cnn(patterns, spec, 2)
    assert omega2[0] != omega2[1], "layer 2 (post-pool) should diverge"
    print("[OK] test_pool_argmax_in_region_id")


def test_chain_advance_matches_from_scratch() -> None:
    """`_advance_chain_state` walked incrementally over layers 1..L must
    produce the same (Ã, c̃, S_l) at every layer as `compute_active_subnetwork_cnn`
    called from scratch at that layer."""
    rng = np.random.default_rng(2)
    spec = LeNetSpec(
        input_shape=(1, 14, 14),
        conv_channels=(3, 4),
        fc_widths=(6, 5),
        kernel_size=3,
        pool_size=2,
        num_classes=4,
    )
    weights = make_random_lenet_weights(spec, rng)
    N = 24
    X = rng.standard_normal((N, spec.input_size)).astype(np.float32)
    patterns = forward_activation_patterns_cnn(spec, weights, X)

    unrolled = [unrolled_conv_W_b(spec, weights, i) for i in range(spec.n_conv)]
    omega_deep = cumulative_pattern_hashes_cnn(patterns, spec, spec.num_relu_layers)
    region_gates = collect_unique_region_gates(patterns, omega_deep)

    for rid, gates in region_gates.items():
        state = _init_chain_state(spec)
        for layer in range(1, spec.num_relu_layers + 1):
            _advance_chain_state(state, spec, weights, gates, layer, unrolled)
            tA, tc, S_l = compute_active_subnetwork_cnn(
                spec, weights, gates, layer, unrolled_convs=unrolled
            )
            assert state.layer == layer
            assert np.array_equal(state.S_prev, S_l), (
                f"S_l mismatch at layer {layer} for rid {rid.hex()[:8]}"
            )
            assert state.tilde_A.shape == tA.shape, (
                f"tilde_A shape mismatch at layer {layer}"
            )
            assert np.allclose(state.tilde_A, tA, atol=1e-5), (
                f"tilde_A mismatch at layer {layer} for rid {rid.hex()[:8]}: "
                f"max|err|={float(np.max(np.abs(state.tilde_A - tA))):.3e}"
            )
            assert np.allclose(state.tilde_c, tc, atol=1e-5), (
                f"tilde_c mismatch at layer {layer} for rid {rid.hex()[:8]}: "
                f"max|err|={float(np.max(np.abs(state.tilde_c - tc))):.3e}"
            )

    # FC-only sanity check: same property must hold with no conv layers.
    spec_fc = LeNetSpec(
        input_shape=(10,),  # type: ignore[arg-type]
        conv_channels=(),
        fc_widths=(7, 5),
        kernel_size=1,
        pool_size=1,
        num_classes=3,
    )
    weights_fc = make_random_lenet_weights(spec_fc, rng)
    X_fc = rng.standard_normal((20, 10)).astype(np.float32)
    patterns_fc = forward_activation_patterns_cnn(spec_fc, weights_fc, X_fc)
    omega_deep_fc = cumulative_pattern_hashes_cnn(
        patterns_fc, spec_fc, spec_fc.num_relu_layers
    )
    region_gates_fc = collect_unique_region_gates(patterns_fc, omega_deep_fc)
    for rid, gates in region_gates_fc.items():
        state = _init_chain_state(spec_fc)
        for layer in range(1, spec_fc.num_relu_layers + 1):
            _advance_chain_state(state, spec_fc, weights_fc, gates, layer, [])
            tA, tc, S_l = compute_active_subnetwork_cnn(
                spec_fc, weights_fc, gates, layer, unrolled_convs=[]
            )
            assert np.array_equal(state.S_prev, S_l)
            assert np.allclose(state.tilde_A, tA, atol=1e-5)
            assert np.allclose(state.tilde_c, tc, atol=1e-5)

    print("[OK] test_chain_advance_matches_from_scratch")


if __name__ == "__main__":
    test_activation_equivalence()
    test_fc_reduction()
    test_pool_argmax_in_region_id()
    test_chain_advance_matches_from_scratch()
    print("All smoke tests passed.")
