# Phase A — detailed implementation plan (baseline comparisons)

Parent: `planning/paper_extensions.md`. This file expands Phase A
into actionable steps: file structure, algorithms with exact
hyperparameters, validation gates, expected outputs, plotting recipes.

The two halves of Phase A are independent and can run in parallel:

- **A.1 — MI baselines** for `Ĩ(Y;Ω)`: MINE, InfoNCE, plug-in
  binning, K-means MI.
- **A.2 — Generalization-gap predictors**: sharpness, spectrally
  normalized margin, path-norm, Frobenius. Compared via Kendall τ
  à la Jiang et al. 2020.

Total dev: ~2 weeks. Total compute: ~10 hours.

---

## A.0 — Shared infrastructure

### A.0.1 — Directory layout

New files only; no modifications to existing modules.

```
src_experiment/baselines/
    __init__.py
    mi_baselines.py         # A.1 estimators
    gen_gap_predictors.py   # A.2 measures
    activations.py          # shared: extract pre/post-activations from HDF5

scripts/
    run_mi_baselines.py
    run_gen_gap_predictors.py
    plot_mi_baselines.py
    plot_gen_gap_predictors.py
    validate_mi_baselines.py        # synthetic-data sanity check

results/
    mi_baselines.csv
    gen_gap_predictors.csv

figures/
    baseline_mi_comparison.png
    baseline_gen_gap_kendall.png

logging/
    YYYY-MM-DD_phase_a_baselines.md
```

### A.0.2 — Shared activation extractor

`src_experiment/baselines/activations.py`:

```python
def load_layer_activations(
    h5_path: Path,
    epoch: int,
    layer: int,                    # 1-indexed; 0 = input
    X: np.ndarray,                 # probe set (already in [-1,1] training-time space)
    kind: Literal["pre", "post"] = "pre",
) -> np.ndarray:
    """Return (N, d_layer) activation matrix for the given layer.

    'pre'  = z = W·a_{l-1} + b   (continuous, signed)
    'post' = a = ReLU(z)         (continuous, non-negative)
    """
```

Implementation: load `W_i, b_i` from
`epochs/epoch_<epoch>/model/`, run a forward pass on `X`, return the
requested layer's pre- or post-activation matrix. Reuse
`forward_activation_patterns` plumbing from `routing_estimator.py`
without modifying it.

This is the single shared piece — both A.1 and A.2 depend on it.

### A.0.3 — Evaluation protocol

Match Exp 2/3 conventions:

- **One row per HDF5.** Deepest layer, last epoch only. (Full
  epoch/layer grid only if a baseline is cheap; MINE is too slow.)
- **Probe**: same probe used in the existing estimator runs
  (composite N=20k, wbc full N=569, mnist test N=10k). Reuse
  `src_experiment/probe_loader.py`.
- **Trustworthy filter**: respect ρ ≤ 0.3 in plotting and headline
  statistics. Compute on all rows; filter at plot time.

---

## A.1 — MI baselines

### A.1.1 — Methods

Each baseline produces a single MI scalar `bits_<method>` per row.

#### MINE (Belghazi et al. 2018, *Mutual Information Neural Estimation*)

- **Estimator**: Donsker-Varadhan,
  `Î(Y;T) = E_{p(y,t)}[T_θ(y,t)] - log E_{p(y)p(t)}[exp T_θ(y,t)]`.
- **Critic Tθ**: 2-layer MLP `(d_T + d_Y_onehot) → 256 → 256 → 1`,
  ReLU, no batchnorm.
- **Joint samples**: `(y_i, t_i)` pairs from probe.
- **Marginal samples**: random shuffle of `y_i` against the same `t_i`.
- **Loss** (gradient-bias-corrected MINE-f variant): use
  `mine-f` (Belghazi App C) — replace `log E[exp T]` with
  `E[exp(T)]/exp(K) - 1 + log(K)` where K is an EMA of `log E[exp T]`.
  EMA decay 0.99.
- **Optimizer**: Adam, lr = 5e-4, β = (0.5, 0.999).
- **Schedule**: 2000 iterations, batch = min(N, 256), 5 critic seeds
  → report mean ± std across seeds.
- **Convergence check**: dump `Î(Y;T)` every 100 iters; if stddev
  over the last 500 iters > 0.05 bits, retrain with double the
  iterations. Log convergence diagnostics to per-row CSV.

Reference implementations consulted:
- `mine-pytorch` GitHub (Schulze, 2019) — ~150 LoC reference.
- Belghazi 2018 supplementary code.

Smoke-test target: see A.1.4.

#### InfoNCE (van den Oord et al. 2018)

- **Estimator**:
  `Î(Y;T) ≥ log K - L_NCE`,
  `L_NCE = -E[log f(y, t) / Σ_{y' in batch} f(y', t)]`.
- **Critic f(y, t)**: bilinear `f(y, t) = exp(y_onehot^T W t)` with
  `W ∈ R^{|Y| × d_T}`. Cleaner than MLP critic for low-dim Y.
- **Optimizer**: Adam, lr = 1e-3.
- **Schedule**: 1000 iterations, batch = 256, 3 seeds.
- **Bound**: log(K) ≈ 8 bits at K=256, larger than typical I(Y;T) on
  these networks, so the bound is not the binding factor.

#### Plug-in binning MI (Tishby-style, Saxe et al. 2018)

- **Quantize** each pre-activation neuron into n_bins=8 uniform
  bins over `[-c_l, c_l]` where `c_l = max_i |z_i|` per layer.
- **Hash** the resulting `(n_bins ** d_l)`-symbol vector per sample
  via MD5-of-packbits (reuse `routing_estimator` machinery).
- **Plug-in** + Miller-Madow on the (binned-pattern, label) joint.
- **Sweep n_bins ∈ {2, 4, 8, 16, 30}** and report per setting.
  Reviewer interest is mostly the n_bins=30 number (close to the
  Saxe et al. baseline).

#### K-means MI

- **Cluster** post-activation matrix `(N, d_l)` with sklearn
  KMeans into K clusters.
- **Plug-in** MI on (cluster_id, label).
- **Sweep K ∈ {|Y|, 2|Y|, 4|Y|, 16, 64, 256}**.
- **Notes**: K = |Y| is a *deliberately-lossy* quotient — the
  natural "your estimator without functional structure" baseline.
  The pitch: ours retrieves more class-info than naive clustering.

### A.1.2 — File structure

`src_experiment/baselines/mi_baselines.py`:

```python
class MINEEstimator:
    def __init__(self, hidden=256, lr=5e-4, n_iter=2000, ema_decay=0.99,
                 device="cpu"): ...
    def estimate(self, T: np.ndarray, Y: np.ndarray,
                 num_classes: int, seed: int) -> dict:
        """Returns {"bits": float, "history": np.ndarray, "wall": float}."""

class InfoNCEEstimator:
    def __init__(self, lr=1e-3, n_iter=1000, batch=256): ...
    def estimate(self, T, Y, num_classes, seed) -> dict: ...

def binning_mi(T: np.ndarray, Y: np.ndarray, n_bins: int) -> dict:
    """Plug-in MI with Miller-Madow on quantised T."""

def kmeans_mi(T: np.ndarray, Y: np.ndarray, K: int, seed: int) -> dict:
    """K-means cluster T into K, plug-in MI on (cluster, Y)."""
```

`scripts/run_mi_baselines.py`:

```
discovers HDF5s (reuse run_label_noise_estimator.discover_jobs API)
for each HDF5:
    load probe (probe_loader)
    extract last-epoch deepest-layer pre-activation T
    run each baseline; collect rows
emit per-HDF5 CSV `<h5>_mi_baselines.csv` with columns:
    network_id, dataset, noise_level/target_dim, arch_str, seed,
    epoch, layer, N, num_classes,
    bits_mine_mean, bits_mine_std,
    bits_infonce_mean, bits_infonce_std,
    bits_binning_2, bits_binning_4, ..., bits_binning_30,
    bits_kmeans_K{|Y|,2|Y|,4|Y|,16,64,256},
    wall_seconds_mine, wall_seconds_infonce,
    bits_ours_raw, bits_ours_func, rho, rho_func   # copied from existing CSV for cross-ref
```

`scripts/run_mi_baselines.py --aggregate` concatenates → `results/mi_baselines.csv`.

### A.1.3 — Hyperparameter notes

- **Probe size truncation**: composite N=20k is large; for MINE,
  subsample to N=5000 per estimate (Belghazi note: MINE is sample-
  efficient, larger N rarely improves the bound). Document in the
  per-row CSV via a `mine_subsample_N` column.
- **Class encoding**: Y as one-hot `R^{|Y|}` for MINE/InfoNCE
  critics; integer labels for binning/k-means.
- **Activation choice**: pre-activation z. The estimator's
  `Ĩ(Y;Ω)` is on the binarized z (sign of z), so this is the
  apples-to-apples comparison. Document in the writeup.

### A.1.4 — Validation (smoke tests before real runs)

`scripts/validate_mi_baselines.py`: synthetic ground-truth check.

1. **Discrete I(X;Y)** with X uniform on {0,...,K-1}, Y = X with
   prob (1-ε), uniform random otherwise. Closed-form
   `I(X;Y) = log K - H((1-ε) δ + ε/K · 1)`. Check:
   - binning_mi(K bins) recovers MI to ±0.05 bits at N=10000.
   - kmeans_mi(K clusters) recovers MI to ±0.05 bits at N=10000.
2. **Gaussian I(X;Y)** with `(X, Y) ~ N(0, [[1,ρ],[ρ,1]])`.
   Closed-form `I(X;Y) = -0.5 log(1-ρ²) / log 2 bits`.
   Check at ρ ∈ {0.3, 0.5, 0.8}, N=5000:
   - MINE recovers MI to ±0.1 bits (mean over 5 seeds).
   - InfoNCE recovers MI to ±0.2 bits (it's a looser bound).
3. **Trivial baseline**: `f(x) = x` should have `I(X;Y)=Y` exactly.

Each test is parametrised; the script asserts on the deltas above
and exits 0 / 1.

This is **the gate**: do not run the full sweep until validation
passes. Empirically MINE often fails on first try (bad lr, too few
iters, broken EMA); the synthetic test catches those.

### A.1.5 — Compute estimate

| step | per-HDF5 cost | total (555 HDF5s) |
|---|---|---|
| extract activations | ~1 s | ~10 min |
| MINE × 5 seeds × 2000 iter | ~25 s (CPU) | **~3.8 h** |
| InfoNCE × 3 seeds × 1000 iter | ~5 s | ~45 min |
| binning × 5 settings | <1 s | ~10 min |
| kmeans × 6 settings | ~3 s | ~30 min |
| **Total A.1** | ~35 s | **~5.5 h** |

CPU is fine; if a GPU is available MINE drops to ~5s/HDF5
(~45 min total).

### A.1.6 — Plotting

`scripts/plot_mi_baselines.py` produces `figures/baseline_mi_comparison.png`:

- **Panel A** (per dataset, 1×3): scatter of `bits_<baseline>` vs
  `bits_ours_raw` (`Ĩ_raw(Y;Ω) + Miller-Madow`) per HDF5,
  colored by baseline method, faceted by dataset. Identity line
  shown. Caption: ours matches MINE within ~0.1 bits across the
  trustworthy regime; binning explodes (DPI violation visible);
  K-means plateaus.
- **Panel B** (1×3): Pearson r of each baseline's MI vs
  `gen_gap_norm` (within-noise stratified). The pitch: ours and
  MINE are competitive; binning/k-means underperform; ours wins on
  cost.
- **Panel C** (single): wall-clock vs accuracy scatter. X-axis
  log-scaled wall, Y-axis match-to-MINE in bits. Ours sits in the
  bottom-left (cheap + accurate); MINE upper-right; binning
  bottom-right (cheap, inaccurate).

### A.1.7 — Risks and mitigations

| risk | mitigation |
|---|---|
| MINE underestimates at low N (Belghazi 2018 fig 4) | Subsample to N=5000 only when N>5000 |
| MINE seed variance large | Report mean ± std across 5 critic seeds; flag any HDF5 with std > 0.2 bits |
| InfoNCE bound saturates at log(K) | Use K=256 batch; saturation is a known limit, document |
| Binning explodes at high n_bins | Stop sweep at n_bins=30; report the curve, not a single number |
| Apples-to-oranges critique (continuous vs discrete) | Run **both** ours-on-quantized (binning at infinitesimal bin-width = quotient) and MINE-on-continuous. The comparison framing is "how close is our discrete summary to the continuous I(Y;T)?" |

### A.1.8 — Deliverables checklist

- [ ] `src_experiment/baselines/mi_baselines.py`
- [ ] `src_experiment/baselines/activations.py`
- [ ] `scripts/validate_mi_baselines.py` passes all assertions
- [ ] `scripts/run_mi_baselines.py` (per-HDF5 CSV emitter)
- [ ] `scripts/run_mi_baselines.py --aggregate` → `results/mi_baselines.csv`
- [ ] `scripts/plot_mi_baselines.py` → `figures/baseline_mi_comparison.png`
- [ ] Writeup: Section 5.1 in `logging/paper_experiments_summary.md`

---

## A.2 — Generalization-gap predictors

### A.2.1 — Methods

Each predictor produces a single scalar per HDF5 (deepest, last
epoch). The pitch: ρ_func at convergence ranks competitively or
better than these standard predictors via Kendall τ (Jiang et al.
2020 protocol).

#### Sharpness (Keskar et al. 2017; Foret et al. 2021 SAM)

- **Definition**: top-k Hessian eigenvalues of training loss at
  the converged model, computed via Lanczos on Hessian-vector
  products.
- **HVP**: `H·v = grad(grad(L, θ).dot(v), θ)`, computed via
  `torch.autograd.grad(create_graph=True)`.
- **Lanczos**: `scipy.sparse.linalg.eigsh` with a `LinearOperator`
  wrapping the HVP, k=5 top eigenvalues.
- **Reduction**: report `λ_max` and `Σ_{i=1..5} λ_i`. Both are
  standard sharpness measures.
- **Subsample for HVP**: full training data is too expensive on
  N=60k MNIST. Subsample 1024 random training points per HVP call
  with a fixed seed (consistent across HVP calls within one
  Lanczos run).
- **Hyperparams**: ncv=20, maxiter=300, tol=1e-3.
- **Per-HDF5 cost**: ~30-60 s on CPU.

#### Spectrally normalized margin (Bartlett et al. 2017; Jiang et al. 2020 simplified)

- **Margin**: `γ_min(X, y) = min_i [f(x_i)_{y_i} - max_{j≠y_i} f(x_i)_j]`
  on the training set.
- **Spectral product**: `Π_l ||W_l||_2` (operator norm via
  `np.linalg.svd(W, compute_uv=False)[0]`).
- **Predictor**: `γ_min / Π_l ||W_l||_2`. Higher = better.
- **Per-HDF5 cost**: ~1 s.

#### Path-norm (Neyshabur et al. 2015)

- **Definition**: `||W||_path = sqrt(Σ_{paths} Π_l w_e²)`. Computed
  by forward pass with squared weights and ones-input.
- **Per-HDF5 cost**: ~1 s.

#### Frobenius

- `||W||_F = sqrt(Σ_l Σ_ij w_lij²)`. Trivial baseline.

#### Final test loss / final test accuracy

Already in HDF5; just read.

### A.2.2 — File structure

`src_experiment/baselines/gen_gap_predictors.py`:

```python
def sharpness(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
              k: int = 5, n_subsample: int = 1024,
              seed: int = 0) -> dict:
    """Returns {"lambda_max": float, "lambda_sum_top5": float, "wall": float}."""

def spectral_margin(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    """Returns {"gamma_min": float, "spectral_prod": float,
                "spectral_margin_ratio": float}."""

def path_norm(model: nn.Module) -> float: ...

def frobenius(model: nn.Module) -> float: ...
```

`scripts/run_gen_gap_predictors.py` discovers HDF5s, rebuilds the
PyTorch model from saved weights via `NeuralNet`, computes each
predictor, emits one row per HDF5 to per-HDF5 CSVs, then aggregates
to `results/gen_gap_predictors.csv`.

### A.2.3 — Comparison protocol (Jiang et al. 2020)

- Compute per-HDF5 predictors plus our `rho_func`.
- For each (dataset × predictor), compute Kendall τ between
  `predictor` and `gen_gap_norm` across all (arch, facet, seed)
  cells, with 2000-resample bootstrap CI.
- Stratify within-noise/PCA (Sec 2.4 protocol).
- Predictors expected to be smaller-is-better (sharpness, path-norm,
  Frobenius) get sign-flipped before correlation so a positive τ
  always means "good predictor".

### A.2.4 — Validation

`scripts/validate_gen_gap_predictors.py`:

1. **Sharpness** on a known toy: a 1D quadratic `L(θ) = aθ²` should
   give `λ_max = 2a`. Check at a ∈ {1, 10, 100}.
2. **Spectral norm** on identity weights: should be 1.0.
3. **Path-norm** on a 2-layer net with all-ones weights: closed
   form. Compare.
4. **Frobenius** on identity matrix: trace check.

These are all closed-form, so the gate is mechanical.

### A.2.5 — Compute estimate

| step | per-HDF5 cost | total (555 HDF5s) |
|---|---|---|
| sharpness (Lanczos) | ~30 s | ~4.6 h |
| spectral margin | ~1 s | ~10 min |
| path-norm + Frobenius | <1 s | ~5 min |
| **Total A.2** | ~35 s | **~5 h** |

### A.2.6 — Plotting

`scripts/plot_gen_gap_predictors.py` produces
`figures/baseline_gen_gap_kendall.png`:

- **Panel A** (1×3, per dataset): bar chart of Kendall τ for each
  predictor (sharpness, spectral_margin, path_norm, frobenius,
  rho_func, rl_proxy) vs gen_gap_norm. Bootstrap 95% CI as error
  bars. Sorted descending by mean τ. Highlights the rank.
- **Panel B** (1×3): same but **within-noise stratified** —
  boxplot showing Kendall τ across each within-noise / within-PCA
  facet. Cleaner test of per-regime predictive power.
- **Output table** (`results/gen_gap_predictors_kendall.csv`):
  one row per (dataset, predictor, subset), columns
  `kendall_tau, lo, hi, n_cells`.

### A.2.7 — Risks and mitigations

| risk | mitigation |
|---|---|
| Sharpness has high variance across seeds | Report median + IQR over 3 evaluation seeds (subsample seed for HVP) |
| Sharpness is undefined on saddle/degenerate Hessians | Catch `eigsh` non-convergence; fall back to `λ_max` from full power iteration |
| Path-norm explodes on deep nets | Use log path-norm (standard) |
| ρ_func wins by accident on a noise-confounded subset | Always report within-noise stratified τ alongside cross-cell τ |
| Different predictors on different scales | Kendall τ is rank-based, scale-invariant — already handles this |

### A.2.8 — Deliverables checklist

- [ ] `src_experiment/baselines/gen_gap_predictors.py`
- [ ] `scripts/validate_gen_gap_predictors.py` passes all assertions
- [ ] `scripts/run_gen_gap_predictors.py` (per-HDF5 + aggregate)
- [ ] `results/gen_gap_predictors.csv`,
      `results/gen_gap_predictors_kendall.csv`
- [ ] `scripts/plot_gen_gap_predictors.py` →
      `figures/baseline_gen_gap_kendall.png`
- [ ] Writeup: Section 5.2 in `logging/paper_experiments_summary.md`

---

## A.3 — Combined Section 5 writeup

After both halves land, write Section 5 in
`logging/paper_experiments_summary.md`:

- **5.0 Setup** — datasets, what columns get added, evaluation
  protocol.
- **5.1 MI estimator comparison** — ours vs MINE/InfoNCE/binning/
  kmeans. Headline: ours within X bits of MINE on average,
  Y-times faster, monotonic across layers (DPI), no
  hyperparameters. Table + Panel A/B/C of `baseline_mi_comparison.png`.
- **5.2 Generalization-gap predictor comparison** — Kendall τ table
  per dataset, plot. Headline: `rho_func` and `rl_proxy` rank in
  the top-K of standard generalization measures across the three
  datasets, especially after within-noise stratification.
- **5.3 What this changes for paper claims** — pre-empts the most
  common reject reasons; explicit positioning vs Jiang et al. 2020
  benchmark; honest acknowledgment of limits.

---

## A.4 — Suggested workflow / week-by-week

| week | A.1 | A.2 |
|---|---|---|
| 1 | Implement `activations.py`, `mi_baselines.py` (binning + kmeans done first; MINE last) | Implement `gen_gap_predictors.py` (Frobenius → path → spectral_margin → sharpness) |
| 1 | `validate_mi_baselines.py` passes | `validate_gen_gap_predictors.py` passes |
| 2 | Run full sweep, aggregate, plot | Run full sweep, aggregate, plot |
| 2 | Section 5.1 draft | Section 5.2 draft |
| 3 | Section 5.3 + figure polish | (slack / buffer for Phase B/C) |

A.1 and A.2 are independent; ideal split for a two-person team
(student + advisor split) or a single-person serial schedule.

---

## A.5 — Branch / git hygiene

Because this introduces a `baselines/` subpackage and several new
scripts, it's natural to do this on a feature branch:

```
git checkout -b phase-a-baselines
# implement, validate, run, plot
git commit -m "Phase A: MI baselines + gen-gap predictors"
# merge to main when Section 5 is drafted
```

Per the project's "no edits to existing modules" principle for the
new estimator work, all new code is *additive* — no existing module
is touched. The only existing-file edit is the new
`logging/paper_experiments_summary.md` Section 5.

---

## A.6 — Stop-the-line conditions

If any of the following come out of the validation gates, halt the
phase and reassess:

1. **MINE fails synthetic Gaussian to ±0.1 bits.** Implementation
   bug; budget another 2 days for debugging.
2. **Sharpness Kendall τ on training loss is ≤ 0 across all
   datasets.** That would suggest the saved model checkpoints
   aren't converged; revisit training.
3. **`rho_func` Kendall τ ranks dead last across all datasets.**
   The ρ_func diagnostic claim is wrong, and Sec 4.5 of the paper
   needs revision before the paper is submission-ready.

The third is the genuine paper-rewriting risk; a graceful response
is to position ρ_func as a *qualitative diagnostic* rather than a
quantitative predictor and emphasize Exp 3 (RL_proxy ↔ quotient
gap) as the main empirical result instead.
