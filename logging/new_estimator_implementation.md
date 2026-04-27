# Routing-information estimator — implementation log

Scope: this document describes the implementation of Recipes 1–4 from
`claude_new_estimator_instructions.md`. All work is in three new modules under
`src_experiment/`; no existing code was modified.

## 1. Where this sits in the existing pipeline

The legacy pipeline is

```
train (PyTorch)  →  HDF5  →  Julia geometric analysis  →  HDF5  →  ExperimentEvaluator (Tree-based MI)
```

The new estimator branches off the HDF5 directly and ignores the Julia tree:

```
train (PyTorch)  →  HDF5  ─┬─────►  ExperimentEvaluator   (legacy, untouched)
                           │
                           └─────►  RoutingEstimator / FunctionalQuotientEstimator (new)
```

It only reads `metadata`, `points`, `labels`, and the per-epoch hidden-layer
weights (`epochs/epoch_<N>/l<i>.weight`, `l<i>.bias`). It does its own
forward pass, so probe sets do not need to be the training-time test set.
This was a deliberate design choice from the prior session's analysis: the
legacy `Tree.perform_number_count` silently drops samples that land in
regions Julia never discovered, which is unsafe for held-out probe sets.

## 2. Module map

| File | Responsibility |
|------|---------------|
| `src_experiment/routing_estimator.py` | Recipe 1. Forward pass, cumulative-pattern md5 hashing, plug-in MI, Miller–Madow correction, ρ, truncation probability. Standalone class `RoutingEstimator`. |
| `src_experiment/functional_quotient.py` | Recipes 2 & 3. Active subnetwork matrix, ε-tolerance clustering, quotient MI. Composes `RoutingEstimator` and the RTG module. Class `FunctionalQuotientEstimator` is the unified driver returning the full row schema. |
| `src_experiment/rtg_analyzer.py` | Recipe 4. Hamming-1 adjacency via single-bit-flip enumeration, union-find connected components, RTG diagnostics. |

Relationship: `functional_quotient` imports from both `routing_estimator` and
`rtg_analyzer`. `routing_estimator` is the only module that touches HDF5;
the other two operate on already-extracted patterns and weights.

## 3. Notation

For a ReLU MLP with $L$ hidden layers and weights $\{W^i, b^i\}_{i=1}^L$:

- Pre-activation at layer $i$: $z^i(x) = W^i a^{i-1}(x) + b^i$, with $a^0 = x$.
- Activation pattern: $\pi^i(x) = \mathbf 1[z^i(x) > 0] \in \{0,1\}^{n_i}$ (strict $>$).
- Cumulative pattern at layer $l$: $\pi^{\le l}(x) = (\pi^1, \pi^2, \dots, \pi^l)$.
- Empirical region: $\omega_i = \pi^{\le l}(x_i)$, with empirical support
  $\Omega_{\mathcal D}^{\le l} = \{\omega_i\}$.
- Active neuron set at layer $i$ for region $\omega$:
  $S^\omega_i = \{j : \pi^i_\omega[j] = 1\}$, with $S^\omega_0 = \{1, \dots, n_0\}$.

## 4. Recipe 1 — routing-information estimator

### Math

Plug-in MI on the contingency table $n_{y,\omega}$:

$$
\hat I(Y; \Omega_{\mathcal D}) = \sum_{y, \omega} \frac{n_{y, \omega}}{N}
  \log_2 \frac{n_{y, \omega} N}{n_y n_\omega}.
$$

Miller–Madow bias correction (the spec's $\ln 2$ converts the nats correction
to bits):

$$
\tilde I = \hat I - \frac{(|\Omega_{\mathcal D}| - 1)(|\mathcal Y| - 1)}{2 N \ln 2}.
$$

Diagnostics:

- $\rho = |\Omega_{\mathcal D}| / N$. The spec's threshold for trustworthiness is
  $\rho \le 0.3$.
- Truncation probability: on a fresh held-out set, fraction of samples whose
  region was not seen in the probe set. Upper-bounds the gap between $\tilde I$
  and the population MI by $H(Y) \cdot \Pr(\Pi \notin \Omega_{\mathcal D})$.

### Implementation

- Forward pass: `forward_activation_patterns(W, b, X)` runs `z = a @ W.T + b`,
  records `pi = z > 0`, then `a = relu(z)`. Uses strict `>` to match the
  mathematical definition.
- Hashing: `cumulative_pattern_hashes(per_layer_patterns, layer)` concatenates
  the per-layer patterns up to `layer`, bit-packs each row with `np.packbits`,
  and digests with `hashlib.md5`. Md5 is deterministic across runs and across
  Python invocations; Python's built-in `hash()` is not.
- Contingency table: built as a `Counter` keyed by `(region_md5, label)`,
  reshaped to a dense `(R, C)` int64 matrix for the MI sum.
- `RoutingEstimator(h5_path)` exposes `evaluate_epoch(...)` and
  `evaluate_all(...)`. Both accept optional `X, y, X_holdout, y_holdout` so
  the user can score arbitrary probe / validation sets.

### Verified invariants (`outputs/blobs_2d/blobs_2d.h5`)

- $H(Y) = 3.3219$ bits = $\log_2 10$ (10-class blobs).
- $\hat I \le H(Y)$ on every row.
- $\tilde I < \hat I$ always (correction is subtractive).
- `num_regions` monotone in layer depth.
- Bit-identical results on repeated runs (md5 determinism).
- Truncation probability rises monotonically with layer depth on a 70/30 split
  (0.6 % → 3.4 % → 7.0 %).

## 5. Recipe 2 — functional-equivalence quotient

### Math

Two regions are functionally equivalent iff their per-region affine maps agree:
$\omega \sim_{\mathrm{func}} \omega'$ iff $A^l_\omega = A^l_{\omega'}$ and
$c^l_\omega = c^l_{\omega'}$.

The full $A^l_\omega \in \mathbb R^{n_l \times n_0}$ has zero rows wherever
$\pi^l_\omega = 0$. Working with the **active subnetwork matrix**

$$
\tilde A^l_\omega = W^l[S^\omega_l, S^\omega_{l-1}] \, W^{l-1}[S^\omega_{l-1}, S^\omega_{l-2}]
  \cdots W^1[S^\omega_1, S^\omega_0],
$$

avoids materializing the zero rows. Building $\tilde A$ and $\tilde c$ together
via the recursion (initialized $\tilde A_0 = I_{n_0}$, $\tilde c_0 = 0$):

$$
\tilde A_i = W^i[S_i, S_{i-1}] \, \tilde A_{i-1}, \qquad
\tilde c_i = W^i[S_i, S_{i-1}] \, \tilde c_{i-1} + b^i[S_i].
$$

This recursion follows from the ReLU forward pass on a region: with
$Q^i_\omega = \mathrm{diag}(\pi^i_\omega)$, we have
$a^i = Q^i_\omega(W^i a^{i-1} + b^i)$, and the active rows of the unrolled
recurrence give $\tilde A_i, \tilde c_i$. (Worth confirming this matches your
main-body Eq. 3 — it is the standard derivation.)

ε-tolerance equivalence:

$$
\omega \sim_\varepsilon \omega' \iff S^\omega_l = S^{\omega'}_l
  \;\wedge\; \|\tilde A^l_\omega - \tilde A^l_{\omega'}\|_F + \|\tilde c^l_\omega - \tilde c^l_{\omega'}\|_2 \le \varepsilon.
$$

The spec writes strict `<`; this implementation uses `<=` so that $\varepsilon = 0$
recovers numerical equality. Otherwise $\varepsilon = 0$ would never cluster
anything.

### Implementation

- `compute_active_subnetwork(W, b, per_layer_patterns, layer)` runs the recursion
  using `np.ix_(S_i, S_{i-1})` for submatrix slicing.
- `_build_active_data(...)` precomputes $(\tilde A, \tilde c, S_l)$ for each unique
  region; reused across all $\varepsilon$ in a sweep.
- `cluster_functional(active_data, eps)` first buckets by $(|S_l|, S_l\text{.tobytes()})$
  — regions with mismatched $S_l$ cannot match by definition. Within each bucket
  it does naive $O(k^2)$ Frobenius+L2 comparison. At observed scales
  (≤ ~5k regions per layer) this is fast; LSH is deferred until needed.

### Verified invariants

- `num_quotient ≤ num_regions` everywhere.
- `num_quotient` monotone non-increasing in $\varepsilon$.
- $\varepsilon = 0$ collapses only numerically equal regions (effectively none, due
  to fp noise).
- Bucketing is non-trivial: at layer 2 of `blobs_2d` (50th epoch), 78 buckets
  hold 164 regions; the largest bucket has 14 members.

## 6. Recipe 3 — quotient MI estimator

### Math

Replace $\Omega_{\mathcal D}^{\le l}$ with $\Omega_{\mathcal D, \mathrm{func}}$ in
Recipe 1's contingency table:

$$
m_{y, [\omega]} = \sum_{\omega' \in [\omega]} n_{y, \omega'}.
$$

Plug-in and Miller–Madow proceed identically with $|\Omega_{\mathcal D, \mathrm{func}}|$
in place of $|\Omega_{\mathcal D}|$. Pattern redundancy ratio
$\rho_{\mathrm{func}} = |\Omega_{\mathcal D, \mathrm{func}}| / |\Omega_{\mathcal D}| \in (0, 1]$.

By the data-processing inequality, $\hat I_{\mathrm{func}} \le \hat I$, with equality
iff every region is in its own equivalence class. The MM correction shrinks
proportionally to the support size, so as $\varepsilon$ grows the *corrected*
$\tilde I_{\mathrm{func}}$ can rise even though the plug-in falls. This is the
intended payoff: a tighter lower bound on $I(Y; \Pi)$.

### Implementation

`routing_information_quotient(omega_ids, y, quotient_map)` is a thin wrapper:
it remaps `omega_ids` through the quotient map and calls the existing Recipe 1
function. ~10 lines.

### Verified invariants

- $\hat I_{\mathrm{func}} \le \hat I$ on every row (DPI).
- $\hat I_{\mathrm{func}}$ monotone non-increasing in $\varepsilon$ (more pooling → less plug-in MI).
- $\tilde I_{\mathrm{func}}$ is non-monotone in $\varepsilon$ — for `blobs_2d`,
  layer 3, epoch 50: 2.155 (ε=0) → 2.546 (ε=10) → 2.720 (ε=100). Good.

## 7. Recipe 4 — data-supported RTG

### Math

Vertices are regions in $\Omega_{\mathcal D}^{\le l}$; edges connect regions
whose cumulative patterns differ in exactly one bit. Spec's faster variant: for
each region, enumerate the $\sum_i n_i$ single-bit flips and probe a hash table
for membership. Cost $O(|\Omega| \cdot \sum_i n_i)$ instead of $O(|\Omega|^2)$.

Diagnostics: number of components, component-size distribution, largest-component
fraction, fraction of regions in size-1 components.

### Implementation

- `cumulative_patterns_per_region(per_layer_patterns, omega_ids, layer)` returns
  `{rid → cumulative bool vector}` for the unique regions only.
- `hamming1_adjacency(region_cum_patterns)` enumerates flips and probes the
  region dict. The region ID **is** the md5 of the bit-packed cumulative
  pattern (by construction in `cumulative_pattern_hashes`), so the same hashing
  is used for the flipped pattern probe — no extra lookup table needed.
- `connected_components` uses a small union-find with path compression; about 25
  lines, no scipy dependency.
- `rtg_diagnostics` returns a dataclass with `num_components`, `component_sizes`
  (sorted desc.), `largest_component_frac`, `isolated_frac`.

### Verified invariants

- Adjacency is symmetric.
- Every edge connects regions whose patterns differ in exactly one bit.
- Components partition the region set (sum of sizes equals `num_regions`).
- On `blobs_2d` epoch 50: layer 1 has 2 components / 45 regions / 96 % giant;
  layer 2 has 11/164/93 %; layer 3 fragments to 33 components with the four
  largest being 71, 65, 52, 43.

## 8. Unified driver

`FunctionalQuotientEstimator(h5_path)` composes `RoutingEstimator` for HDF5
loading and runs all four recipes per epoch and per layer:

```
for epoch in checkpoints:
    W, b ← load weights from HDF5
    per_layer_patterns ← forward_activation_patterns(W, b, X)
    for layer in 1..L:
        omega ← cumulative_pattern_hashes(per_layer_patterns, layer)
        Recipe 1 → (plug_in, MM, R, H_Y, ρ, truncation_prob)
        Recipe 4 → (num_components, sizes, fractions)         # ε-independent
        for ε in epsilons:
            Recipe 2 → (quotient_map, num_quotient)
            Recipe 3 → (plug_in_func, MM_func)
            yield row(epoch, layer, ε, ...)
```

Recipe-2 active-data is computed once per (epoch, layer) and reused across the
ε sweep. Recipe 4 is computed once per (epoch, layer) and duplicated across ε
in the output rows (it does not depend on ε).

## 9. Output schema

One row per `(network_id, epoch, layer, ε)`:

```
network_id, epoch, layer, epsilon, seed, N,
num_regions, num_quotient,
rho, rho_func,
plug_in_bits, miller_madow_bits,
plug_in_func_bits, miller_madow_func_bits,
H_Y_bits, truncation_prob,
num_rtg_components, rtg_largest_component_frac, rtg_isolated_frac
```

The columns required by the spec output schema are all present. The few extras
(`rho`, `rtg_largest_component_frac`, `rtg_isolated_frac`) are diagnostic.

## 10. How to call it

```python
from src_experiment.functional_quotient import FunctionalQuotientEstimator

est = FunctionalQuotientEstimator("outputs/blobs_2d/blobs_2d.h5")

# Default: stored test set as probe, no holdout, spec's ε grid.
df = est.evaluate_all()

# With a fresh probe + holdout for truncation probability + custom ε sweep.
df = est.evaluate_all(
    X=X_probe, y=y_probe,
    X_holdout=X_val, y_holdout=y_val,
    epsilons=(0.0, 1e-2, 1.0, 10.0, 100.0),
)
```

CLI (per module):

```
uv run python -m src_experiment.routing_estimator    outputs/blobs_2d/blobs_2d.h5
uv run python -m src_experiment.functional_quotient  outputs/blobs_2d/blobs_2d.h5
uv run python -m src_experiment.rtg_analyzer         outputs/blobs_2d/blobs_2d.h5 50
```

## 11. Empirical findings to remember

1. **The spec's default ε grid is too small for trained MLPs.** With
   `{0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1}`, no regions ever collapse on
   `blobs_2d`. The natural functional-separation scale lives near
   ε ≈ 10–100; collapse begins around ε = 10 (≈ 50 % at layer 3) and reaches
   the bucket-floor near ε = 100. The spec's range catches floating-point
   noise; the empirical separation between distinct trained-network regions
   is orders of magnitude larger than that. You should sweep wider.

2. **Recipe 3's Miller–Madow MI rises with ε.** For `blobs_2d`, layer 3,
   epoch 50: 2.155 (ε=0) → 2.546 (ε=10) → 2.720 (ε=100), even though the
   plug-in MI falls. This is the DPI payoff and the reason for using the
   quotient at all.

3. **ρ stays well below 0.3 on `blobs_2d`** at N=2000 (max 0.14 at layer 3,
   epoch 50). Other experiments — particularly higher-dim blobs — were
   flagged in the prior analysis as having ρ > 0.3. Those experiments need
   bigger probe sets before the estimator is trustworthy on them.

4. **RTG fragments with depth.** Component count grows 2 → 11 → 33 across the
   three hidden layers, while the giant-component fraction shrinks
   96 % → 93 % → 26 %. The cumulative pattern at layer 3 has 75 bits, so
   single-bit-distance neighbours are sparser; the data-supported RTG no
   longer behaves like a near-complete subgraph.

## 12. Known gaps / open items

- **Unapplied MM correction in `ExperimentEvaluator._compute_region_counts:271`.**
  `MMcorr` is computed and stored as a diagnostic column, but the reported
  `I(Y;W)` is the uncorrected plug-in. Trivial one-line fix; left untouched
  per "do not edit existing files" rule, but should be addressed for
  consistency between the legacy and new pipelines.
- **Bias recursion vs. main-body Eq. 3.** The recursion used here
  ($\tilde c_i = W^i[S_i, S_{i-1}] \tilde c_{i-1} + b^i[S_i]$) is the standard
  derivation from $a^i = Q^i_\omega (W^i a^{i-1} + b^i)$. Worth a quick
  side-by-side check with the equation as written in the paper.
- **LSH for Recipe 2.** Naive $O(R^2)$ within-bucket comparison is fine at
  observed scales (≤ ~5k regions per layer). For ρ > 0.3 experiments that
  scale up the probe set, the spec recommends LSH on rounded $\tilde A$.
- **Truncation probability requires a fresh validation set.** The HDF5 only
  stores one test split, so any disjoint validation set must be supplied
  externally (e.g. by re-generating from the same dataset config with a
  different `global_seed`).
- **Recipe 2 / 3 interaction with $\varepsilon = 0$.** The spec writes strict
  `<`; this implementation uses `<=` so $\varepsilon = 0$ recovers numerical
  equality. With float32 weights, even truly-equivalent regions will rarely
  produce bit-identical $\tilde A$ matrices, so $\varepsilon = 0$ effectively
  reports `num_quotient = num_regions`. The plateau-finding workflow in the
  spec is the right way to read the sweep.

## 13. Driver, probe-loader, and plotting (label-noise rollout)

Tooling for running the estimator over the existing label-noise sweeps lives at
the repo root and under `visualization/`:

- **`run_label_noise_estimator.py`** — driver over
  `outputs/{composite,wbc}_label_noise/`. Walks `n<noise>_<arch>/seed_<seed>.h5`,
  evaluates each via `FunctionalQuotientEstimator.evaluate_all`, writes
  `new_estimator_seed_<seed>.csv` next to the HDF5 (idempotent: skip if exists).
  Per-job timing + ETA. CLI flags filter by dataset / noise / arch / seed and
  control the probe policy. `--aggregate` concatenates everything into a single
  CSV with `dataset`, `noise_level`, `arch_str`, `probe_N` columns added.

- **`src_experiment/probe_loader.py`** — `lru_cache`'d probe + holdout builders.
  - `make_composite_probe(global_seed=42, probe_size, holdout_size, ...)`:
    rebuilds the training-time `MinMaxScaler` by replaying
    `_make_composite_data(N_SAMPLES, seed=42)` → 80/20 stratified split → fit;
    then `_make_composite_data` is called again with fresh seeds and pushed
    through the same scaler. Labels stay clean (label noise was injected on
    train labels only — for routing analysis we want $I(Y_{\text{true}};\Omega)$).
  - `make_wbc_probe(mode={"test","full","split"})`: `test` uses the stored
    114-sample test set; `full` uses the entire 569-sample UCI WBC (the train
    portion is "seen" by the model but routing analysis stays well-defined);
    `split` does a 70/30 stratified cut of the stored test set.

- **`visualization/plot_new_estimator.py`** — single CSV in, figures out.
  - **Single-experiment dashboard** (default): 2 × 3 panel — MI trajectories
    at chosen ε (plug-in / MM / MM-quotient per layer), ρ trustworthiness
    check, ρ_func collapse curve at one epoch, RTG fractions over epochs,
    truncation probability over epochs, DPI scatter (plug-in vs quotient).
  - **Noise-compare** (`--noise-compare --dataset --arch`): operates on the
    aggregated CSV. 2 × 2 panel of `miller_madow_bits`, `miller_madow_func_bits`,
    `rho_func`, and `rtg_largest_component_frac`, x = epoch, hue = noise level,
    mean ± std band over seeds at fixed ε / layer.

Verified end-to-end on `composite/n0.0/[25, 25, 25, 25, 25]/seed_101` with
N_probe = 20000, N_holdout = 10000: ρ ≤ 0.106 across all 5 layers,
truncation_prob ∈ [0.0008, 0.027], DPI holds everywhere, 59 s wall-time
(880 rows = 22 epochs × 5 layers × 8 epsilons).

## 14. Reference points

- Original spec: `claude_new_estimator_instructions.md` (in repo root).
- Six-question analysis from the prior session: `logging/compatability_analysis.md`.
- Codebase entry points (legacy, untouched):
  - `src_experiment/estimate_quantities.py` — `ExperimentEvaluator`.
  - `geobin_py/reconstruction.py` — `Tree`, `Region`.
  - `src_experiment/utils.py` — `NeuralNet` (PyTorch model).
