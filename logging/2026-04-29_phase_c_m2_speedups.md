# 2026-04-29 — Phase C-M2 speedups + correctness gates

Continuation of Phase C-M2 (CNN estimator). Phase C-M1 (full-MNIST
LeNet5 training, 4 archs × 3 noise × 5 seeds = 60 HDF5s) had landed in
the previous session, and Phase C-M2 had a working
`src_experiment/cnn_estimator.py` plus three smoke tests. This session
addressed the binding bottleneck flagged at the end of M2: a single
`evaluate_epoch` on full N=10 000 probe took 30+ min before being
killed, with the active-subnetwork chain being re-walked from scratch
per layer.

## What landed

### Single-walk amortization (`src_experiment/cnn_estimator.py`)

- New `_ChainState` dataclass (`tilde_A`, `tilde_c`, `S_prev`, `layer`)
  carries the per-region active-subnetwork build forward across layers.
- `_init_chain_state`, `_apply_conv_relu`, `_apply_pool`,
  `_apply_fc_relu`, `_advance_chain_state(state, target_layer)` —
  forward-only incremental advancement; one ReLU step per `+1` of
  `target_layer`.
- `FunctionalQuotientEstimatorCNN.evaluate_epoch` rewritten:
  region IDs are computed once at the deepest layer (cumulative pattern
  is a prefix); each shallow-layer rid is mapped to a deepest-layer
  representative whose gates' prefix realizes that shallow rid; per-rid
  `_ChainState` is lazily allocated on first lookup (eager-init was a
  ~25 GB blowup on full MNIST: 10 000 × `eye(784)` ≈ 25 GB).
- Net effect: each region's active-subnetwork chain is walked exactly
  once per epoch instead of `L` times — an `L`-fold reduction in the
  dominant `compute_active_subnetwork_cnn` work, where `L = n_conv +
  n_fc_hidden = 4` for LeNet-S.

### Stratified probe subsample

- `_stratified_subsample_indices(labels, n, rng)` — proportional class
  rounding, fractional-remainder allocation for the deficit, leftover
  redistribution if any class is undersized.
- `FunctionalQuotientEstimatorCNN(probe_subsample=N, probe_seed=k,
  stratify=True)` — class-stratified subsample at construction time.
  No subsample → full probe. `subsample_indices` is exposed for
  hold-out alignment downstream.

### Correctness gate (`tests/test_cnn_estimator.py`)

- New `test_chain_advance_matches_from_scratch`: for both a real LeNet
  (input 14×14, conv (3,4), kernel 3, pool 2, fc (6,5)) and the FC-only
  reduction (no conv layers), iterate layers `1..L` advancing a single
  `_ChainState`, and assert `(Ã, c̃, S_l)` matches
  `compute_active_subnetwork_cnn` from scratch at every layer.
- This test caught one real bug en route: when `spec.n_conv == 0`,
  `_advance_chain_state` was unconditionally applying a pool at index
  `-1` before the first FC layer because the "cross final pool" guard
  read `state.layer == spec.n_conv` (true for `n_conv == 0`,
  `state.layer == 0`). Fixed by adding `and spec.n_conv > 0` at
  `cnn_estimator.py:587`.

All four smoke tests pass:

```
[OK] test_activation_equivalence
[OK] test_fc_reduction
[OK] test_pool_argmax_in_region_id
[OK] test_chain_advance_matches_from_scratch
```

## Timing on real LeNet-XS (mnist_full, conv (4,8), fc (60,42))

Single epoch (`epoch=10`) on `outputs/mnist_full_lenet/n0.0_LeNet-XS/seed_101.h5`,
all four ReLU layers, with `probe_subsample=2000` (stratified):

| ε grid                          | wall   | rows |
|---|---:|---:|
| `(0, 1e-2, 1.0)`                | 77.2 s | 12   |
| `(0, 1, 10, 100, 1000)`         | 117.4 s| 20   |

Down from the prior 30+ min hang at full N=10 000 with 3 ε's. The
remaining cost is the per-ε Frobenius-bucket O(R²) clustering, which
scales as `R² · |S_l|` and is now the binding operation rather than
the active-subnetwork build.

## ε-scale finding on real MNIST images

The default ε grid `{0, 1e-8, …, 1e-1}` from the spec doesn't merge
anything on full-MNIST LeNet-XS — same story as on the MLPs (memory:
`project_routing_estimator.md`). With a wider grid, clustering finally
exercises:

| layer | ε=0 | ε=1 | ε=10 | ε=100 | ε=1000 |
|---:|---:|---:|---:|---:|---:|
| 1 (conv) | 2000 | 2000 | 2000 | 2000 | 2000 |
| 2 (conv) | 2000 | 2000 | 2000 | 2000 | 2000 |
| 3 (fc)   | 2000 | 2000 | 1998 | 1967 | 1967 |
| 4 (fc)   | 2000 | 2000 | 2000 | 1780 | 1780 |

Conv layers stay fully saturated even at ε=1000 — the conv-`Ã`
matrices live in a higher-dimensional space and the natural Frobenius
scale there is much larger (or scale-free Frobenius isn't the right
metric for them). The deepest FC drops from 2000→1780 quotient regions
at ε=100, with `mm_func_bits` recovering from −3.17 → −2.46 — the DPI
payoff predicted by Prop 4.6.

For Phase C-M3, a sensible ε grid is `{0, 1, 10, 100, 1000}`. Expect
ρ_func ≈ 1 on conv layers regardless; the headline plot will live on
the deepest FC.

## Status going into M3

- Estimator code path: ✓ (single-walk + lazy chain_states + subsample).
- Correctness gates: ✓ (4 smoke tests, including chain-advance against
  the from-scratch baseline).
- Timing budget: 60 HDF5s × ~17 saved epochs × ~2 min/epoch ≈ ~30 hours
  at N=2000, 5 ε's, 4 layers. Roughly matches Phase C plan §C.4.
- Sweep wrapper: not yet written. Mirrors `run_label_noise_estimator.py`
  with the new driver — to be added in M3.

## Files touched

- `src_experiment/cnn_estimator.py` — `_ChainState` family,
  `_advance_chain_state`, `_stratified_subsample_indices`,
  `FunctionalQuotientEstimatorCNN.{__init__, evaluate_epoch}`
  rewritten around lazy chain_states.
- `tests/test_cnn_estimator.py` — new
  `test_chain_advance_matches_from_scratch`.
- Memory: `project_routing_estimator.md` updated with timing,
  ε-scale observations, and the bug fix.
