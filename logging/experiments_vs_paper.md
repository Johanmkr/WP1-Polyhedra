# Experiments overview vs the NeurIPS 2026 draft

Last update: 2026-04-29.

This file is the cross-reference between every experimental result
the project has produced and the NeurIPS-2026 paper draft
(`Johan_binning_paper_NeurIPS_2026-1.pdf`). Each row says: *what
section/proposition the experiment supports*, *what the headline
finding is*, and *where the numbers/figures live*.

The paper makes three contributions (Abstract / §1):

1. **Routing information** `I(Y;Π)` as the partition-aware MI term
   (Theorem 4.1, Theorem 4.5).
2. **Partition-aware estimator + functional-equivalence quotient**
   (`Ĩ(Y;Ω_D)` and `ρ_func`, §4.5; **Prop 4.6**).
3. **Topological analysis via the data-supported RTG** (§3.2; the
   `RL_proxy` surrogate for **Prop 4.2**).

The experiment series below is organized to land each contribution
empirically.

---

## Experiment 1 — Training dynamics of `ρ_func`
**Anchors:** §4.5 (functional-equivalence quotient as a discrete
IB-analog).
**Status:** ✓ done.

| dataset    | what we see                                                     | reference figure |
|---         |---                                                              |---               |
| composite  | `[7,7,7]` clean: ρ_func 0.107 → 0.083 (e=2) → 0.718              | `figures/training_dynamics/composite_*.png` |
| composite  | `[25,25,25]` clean→noisy: ρ_func 0.40 → 0.23 — **noise suppresses the expansion phase** | `figures/training_dynamics/composite_*.png` |
| wbc        | visible on `[25,25,25]` clean (0.355 → 0.230 → 0.240); shallow recovery (N=569) | `figures/training_dynamics/wbc_*.png` |
| mnist (PCA)| expansion strength scales with PCA target_dim                    | `figures/training_dynamics/mnist_*.png` |

**Headline.** ρ_func at the deepest layer follows a
*compression-then-expansion* trajectory across epochs — the discrete
analogue of the IB compression/fitting phases. **Label noise
suppresses the expansion phase.** Combined with the simultaneous
*rise* in `Ĩ_func` under noise, this is read as
**memorization-via-redundant-routing**: many structurally distinct
polytopes carved around noisy points, but most collapse onto the same
affine map (Prop 4.6's quotient is the active object).

Detailed log: `logging/2026-04-27_paper_grounded_experiments.md`.
Cross-dataset replication: §4 of `logging/paper_experiments_summary.md`.

---

## Experiment 2 — `ρ_func` ↔ generalization gap
**Anchors:** §4.5 (Prop 4.6 — quotient as the right discrete capacity
proxy).
**Status:** ✓ done.

Setup: deepest layer at ε = 10, last epoch, mean over seeds per
(arch, facet) cell. Two gap metrics: `gen_gap_norm = train_acc −
(1−noise)·test_acc` and `gen_gap_abs = |train − test|`. Pearson r
with 95% bootstrap CI (2000 resamples).

**Cross-cell:**

| dataset    | metric        | n   | r [95% CI]                |
|---         |---            |---:|---                          |
| composite  | gen_gap_abs   | 36 | **−0.86** [−0.93, −0.79]    |
| wbc        | gen_gap_norm  | 54 | +0.31 [−0.07, +0.62]        |
| mnist      | gen_gap_norm  | 24 | **+0.81** [+0.71, +0.90]    |

**Within-noise (the rescued result):**

| dataset    | facet                | n   | r [95% CI]                |
|---         |---                   |---:|---                          |
| wbc        | noise = 0.0          | 18 | **+0.61** [+0.24, +0.86]    |
| wbc        | noise = 0.2          | 18 | **+0.62** [+0.05, +0.87]    |
| wbc        | noise = 0.4          | 18 | **+0.80** [+0.52, +0.94]    |
| mnist      | target_dim = 2..20   |  6 | +0.73 .. +0.86 across PCA   |
| composite  | every noise level    | 12 | saturated (`gen_gap_norm` ≈ 0)|

**Reading.** Cross-cell composite r = −0.86 was a **noise-axis
confound**, not the predicted positive correlation; stripping that
confound (within-noise stratification) restores the positive sign on
wbc and mnist. Composite at noise=0 is too easy to expose the gap
dynamic. The paper-claim that survives is **per-regime, not
cross-regime**: at fixed label noise / fixed PCA, more combinatorial
capacity used → wider gap.

Outputs: `figures/rho_func_vs_gen_gap.png`,
`results/rho_func_vs_gen_gap_correlations.csv`,
`results/rho_func_vs_gen_gap_within.csv`. Driver:
`scripts/rho_func_vs_gen_gap.py`. Detailed log: §2 of
`paper_experiments_summary.md`.

---

## Experiment 3 — `RL_proxy` ↔ quotient gap (Prop 4.2 surrogate)
**Anchors:** **Proposition 4.2** (routing loss `I(Y;Π|T)` is positive
iff there exists a Hamming-1-adjacent pair of regions in the
data-supported RTG whose dominant classes disagree); §3.2 (RTG).
**Status:** ✓ done. **Strongest single result of the series.**

Setup: `RL_proxy` is the empirical fraction of Hamming-1-adjacent
region pairs with disagreeing dominant class; `Ĩ_raw − Ĩ_func`
(Miller-Madow corrected) is the finite-data upper-bound surrogate on
the routing-loss term (§4.5).

**Cross-cell Pearson r [95% CI]:**

| dataset    | subset       | n   | r [CI]                  |
|---         |---           |---:|---                       |
| composite  | all          | 36 | **+0.77** [+0.60, +0.87] |
| wbc        | trustworthy  | 25 | **+0.83** [+0.73, +0.94] |
| mnist      | all          | 21 | **+0.68** [+0.37, +0.84] |

**Within-noise (Exp 3 strengthens, not weakens):**

| dataset    | facet                | n   | r [95% CI]                |
|---         |---                   |---:|---                          |
| composite  | noise = 0.0          | 12 | **+0.87** [+0.70, +0.98]    |
| composite  | noise = 0.2          | 12 | **+0.91** [+0.80, +0.97]    |
| composite  | noise = 0.4          | 12 | **+0.93** [+0.90, +0.98]    |
| wbc        | noise = 0.4          | 18 | **+0.80** [+0.68, +0.93]    |

**Reading.** Across **every dataset and trust subset**, RL_proxy is
strongly positively correlated with the quotient gap, with bootstrap
CIs entirely above zero. The composite within-noise result at
`noise = 0.4` (r = +0.93 [+0.90, +0.98], n = 12) is the **cleanest
empirical confirmation of Prop 4.2** in the data and the headline
candidate for §6 of the paper.

Outputs: `figures/rl_proxy_vs_quotient_gap.png`,
`results/rl_proxy_vs_quotient_gap_correlations.csv`,
`results/rl_proxy_vs_quotient_gap_within.csv`. Driver:
`scripts/rl_proxy_vs_quotient_gap.py`. Code:
`src_experiment/rtg_overlap.py` (`region_dominant_class`,
`routing_loss_proxy`). Detailed log: §3 of
`paper_experiments_summary.md`.

---

## Phase A.1 — MI baseline comparison
**Anchors:** §1 reviewer pre-emption ("why not MINE?"); §5 baseline
table.
**Status:** ✓ done.

Five baselines compared against ours (`Ĩ_raw`, `Ĩ_func`):
`binning` (per-neuron uniform plug-in), `kmeans` (cluster-MI at
`K=|Y|`), `KSG` (Ross 2014 mixed continuous-discrete), `InfoNCE`
(bilinear critic), `MINE-f` (Donsker-Varadhan with EMA bias
correction). Validated against synthetic / closed-form ground truth
(12 assertions; max |Δ| ≤ 0.021 bits).

**Cost-accuracy frontier (median over 450 rows; |Δ| vs MINE-f):**

| baseline       | wall (s, composite) | |Δ vs MINE| (bits) |
|---             |---:                 |---:                |
| binning_8      | 0.016               | 0.320              |
| kmeans_\|Y\|   | 1.235               | 0.554              |
| **KSG_k3**     | **0.061**           | **0.068**          |
| InfoNCE        | 1.046               | 0.384              |
| MINE-f (ref)   | 24.96               | 0.000              |

KSG dominates the baseline frontier (~400× cheaper than MINE,
~0.07 bits off on composite). `Ĩ_ours,func` tracks MINE within ~0.1 r
on the high-signal slice (composite n ≥ 0.2) and ties MINE/KSG on
wbc n=0.4, with no tunable hyperparameter.

Outputs: `results/mi_baselines.csv`,
`results/baseline_mi_summary.csv`,
`figures/baseline_mi_{comparison,panel_b,vs_noise}.png`. Code:
`src_experiment/baselines/{activations,mi_baselines}.py`. Detailed
log: §5.1 of `paper_experiments_summary.md`.

---

## Phase A.2 — Generalization-gap predictor comparison
**Anchors:** §5.2 — Jiang et al. 2020 protocol against ρ_func.
**Status:** ✓ done.

Four standard predictors (`sharpness λ_max`, `log path-norm`,
`Frobenius`, `spectral_margin`) vs ours (`ρ_func`, `rl_proxy`,
`Ĩ_raw`). Validated against closed-form / brute-force ground truth
(7 assertions; sharpness Σ top-5 < 1e-2 vs full Hessian on a
27-param net).

**Cross-cell Kendall τ at deepest layer × ε=10 × last epoch:**

| predictor         | composite (n=180)     | wbc (n=270)         |
|---                |---:                   |---:                  |
| **ρ_func (ours)** | **+0.57** [.52, .62]  | +0.11 [.02, .18]     |
| log path-norm     | +0.47                 | +0.34                |
| Frobenius         | +0.27                 | +0.19                |
| sharpness λ_max   | +0.26                 | **+0.43** [.36, .49] |
| rl_proxy (ours)   | +0.16                 | −0.33                |
| Ĩ_raw (ours)      | +0.13                 | −0.19                |

`ρ_func` is the **top cross-cell predictor on composite**, beating
all standard baselines. On wbc, sharpness λ_max wins and `ρ_func`
sits mid-pack. `rl_proxy` and `Ĩ_raw` flip sign across datasets and
are positioned in the paper as **per-regime diagnostics**, not
universal Jiang-style predictors. Frobenius's high-noise sign flip on
composite (`τ = +0.27` cross-cell, `τ = −0.68` at n=0.4 within-noise)
is the cleanest example of why within-noise stratification is the
right protocol.

Outputs: `results/gen_gap_predictors.csv`,
`results/gen_gap_predictors_kendall.csv`,
`figures/baseline_gen_gap_kendall.png`. Code:
`src_experiment/baselines/gen_gap_predictors.py`. Detailed log:
§5.2 of `paper_experiments_summary.md`.

---

## Phase B — Larger tabular UCI
**Status:** ✗ not yet started. Phase A took precedence; Phase B is the
cheapest pending extension and is intended to add a 2–5k-sample UCI
dataset with 5–10 classes so Exp 2.4 can land with tighter CIs (wbc's
N=569 is the binding limit on the within-noise correlations).

---

## Phase C — Full MNIST + small CNN
**Anchors:** §6 of the paper draft (planned). Closes the
"doesn't scale to real data" and "doesn't scale beyond MLPs"
critiques.
**Status:** **partial.** C-M1 ✓ training, C-M2 ✓ estimator + smoke
tests + speedups, **C-M3 sweep not yet launched**.

### C-M1 — training (✓ done)

- `src_experiment/utils.LeNet5(nn.Module)` parametric on
  `conv_channels`, `fc_widths`, `kernel_size`, `pool_size`,
  `input_shape`.
- 4 archs (LeNet-XS / S / M / L) × 3 noise (0, 0.2, 0.4) × 5 seeds
  (101–105) = **60 trained HDF5s** under
  `outputs/mnist_full_lenet/`.
- Driver: `run_mnist_full_lenet_sweep.sh`,
  `scripts/generate_mnist_full_lenet_configs.py`.

### C-M2 — estimator extension (✓ done)

- `src_experiment/cnn_estimator.py` — full LeNet5 estimator analogue:
  `LeNetSpec`, `forward_activation_patterns_cnn`, sparse unrolled
  conv, per-region pool argmax selectors,
  `FunctionalQuotientEstimatorCNN`. Cumulative pattern at layer `l`
  includes both ReLU bools and pool argmax ints crossed up to `l`;
  RTG is built on ReLU bits only.
- Smoke tests in `tests/test_cnn_estimator.py`: activation
  equivalence (`Ã x + c̃` = live hidden activation), FC reduction
  (CNN with `conv_channels=()` matches the FC pipeline), pool-argmax
  separability, and chain-advance matches from-scratch.
- Speedups (this session, 2026-04-29): single-walk
  `_advance_chain_state` (L-fold reduction), lazy chain_states (avoids
  ~25 GB eager-allocation blowup), stratified probe subsample.
- Timing on LeNet-XS @ epoch=10, N=2000: **77 s** for 3 ε's, **117 s**
  for 5 ε's across all 4 layers — down from a 30+ min hang at full
  N=10 000.
- ε-scale observation: FC layers cluster at ε ≥ 10; conv layers stay
  saturated even at ε=1000 (high-dim conv-`Ã` space).

Detailed log: `logging/2026-04-29_phase_c_m2_speedups.md`. Memory:
`project_routing_estimator.md`.

### C-M3 — sweep + paper §6 (✗ not yet done)

Pending work:

- Wrapper script (`run_mnist_full_lenet_estimator.sh` + a
  `run_mnist_full_lenet_estimator.py` analogue of
  `run_label_noise_estimator.py`) to drive the 60 HDF5s through
  `FunctionalQuotientEstimatorCNN` with `probe_subsample=2000`,
  ε grid `{0, 1, 10, 100, 1000}`, all 4 layers.
- Compute budget: 60 × ~17 saved epochs × ~2 min ≈ **~30 hours** on
  one machine (~Phase C plan §C.4).
- Outputs to write: `results/mnist_full_lenet_new_estimator.csv`,
  `figures/training_dynamics/mnist_full_lenet_*.png`,
  `figures/rho_func_vs_gen_gap_full_mnist.png`,
  `figures/rl_proxy_vs_quotient_gap_full_mnist.png`.
- §6 of `paper_experiments_summary.md`: replication of Exp 1, 2.4, 3
  on full MNIST + LeNet, with the noise=0.4 slice as the headline
  (the §C.6 risk #3 — clean MNIST gap is too small).
- Phase A on mnist (deferred per §5.0): rerun MI baselines and
  gen-gap predictors against the LeNet runs once C-M3 lands, so
  the §5 baseline table includes the CNN row.

---

## Cross-dataset replication scorecard (from §4 of `paper_experiments_summary.md`)

| #  | Finding                                                          | composite | wbc | mnist (PCA) |
|---|---                                                                |---:|---:|---:|
| 1 | Compression-then-expansion in ρ_func across epochs                | ✓  | ◐  | ✓  |
| 2 | Label noise suppresses the expansion phase                        | ✓  | ◐  | —  |
| 3 | Cross-cell ρ_func ↔ gen_gap_norm                                  | ✗ (confounded) | ∅ | ✓ |
| 4 | Cross-cell ρ_func ↔ \|train−test\|                                | ✗ (confounded) | ∅ | ∅ |
| 5 | Within-noise ρ_func ↔ gen_gap_norm                                | ∅ (saturated) | ✓ | ✓ |
| 6 | RL_proxy ↔ Ĩ_raw − Ĩ_func                                         | ✓  | ✓  | ✓  |
| **Score** | (✓ + ½·◐ out of 6)                                       | **3.5** | **3.5** | **4.5 / 5** |

- **Exp 3 replicates everywhere** — the one paper-grade finding that
  is significantly positive on every dataset and every trust subset.
- **mnist** is the strongest paper anchor (4.5 / 5 applicable findings
  replicate). It is also the natural Phase C anchor: full-MNIST LeNet
  is the next replication target.
- **composite** is the strongest *manipulation* dataset; **wbc**
  carries the cross-task replication weight despite N = 569.

---

## Open follow-ups carried forward

1. **Phase C-M3 sweep** — see C-M3 above. The remaining gate to a
   paper §6.
2. **`comp_new_lf_*` ablation** — apply Exp 1–3 to the GeoLoss-trained
   composite runs in `outputs/comp_new_lf_*`. The quotient MI should
   separate GeoLoss from vanilla by construction.
3. **MNIST label-noise training sweep** — composite's
   memorization-via-redundant-routing finding hinges on the noise
   axis. Replicating it on (PCA) MNIST needs fresh training at noise
   ∈ {0, 0.2, 0.4} on `[5,5,5]` / `[7,7,7]` × PCA ∈ {5, 20}.
4. **Densify `[7]³`-region on MNIST** — widths {6, 7, 8, 9, 10, 12} ×
   PCA ∈ {5, 10, 20} would give a publication-quality capacity curve.
5. **Larger UCI dataset (Phase B)** — see Phase B above.
6. **LSH for Recipe 2** — current ε > 0 Frobenius-bucket clustering is
   O(R²·|S_l|) and is the binding cost on full MNIST. Becomes a real
   ask if N=2000 isn't enough for §6 figures.
