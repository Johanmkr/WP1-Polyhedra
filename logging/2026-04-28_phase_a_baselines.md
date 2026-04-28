# 2026-04-28 — Phase A baselines (A.1 + A.2) implementation

Implements `planning/phase_a_baselines.md` §A.0 (shared activation
extractor), §A.1 (five MI baselines), and §A.2 (four gen-gap
predictors), each with a synthetic / closed-form validation gate and a
per-HDF5 sweep runner. A unified shell driver `run_phase_a_baselines.sh`
chains validate → sweep → aggregate for both halves so the entire
Phase A can run with one command later.

## What landed

### Code

- `src_experiment/baselines/__init__.py` — package marker.
- `src_experiment/baselines/activations.py` — `load_layer_activations(h5,
  epoch, layer, X, kind)` reused by both A.1 and A.2. Layer 0 returns the
  input; layers 1..L return pre/post activations matching the strict
  `z > 0` ReLU convention used by `routing_estimator.forward_activation_patterns`.
  Output layer (L+1) treats post == pre (no nonlinearity).
- `src_experiment/baselines/mi_baselines.py` — five estimators, each
  returning `{"bits": …, "wall": …, …}`:
  - `binning_mi` — per-neuron uniform quantize over `[-c_l, c_l]`,
    md5-hash, plug-in MI with Miller-Madow via
    `routing_estimator.routing_information`.
  - `kmeans_mi` — sklearn KMeans cluster IDs as the discrete summary,
    plug-in MI on (cluster, Y).
  - `ksg_mi` — KSG / Ross 2014 mixed continuous-discrete, sklearn
    KDTree (Chebyshev), drops singleton-label samples to mirror
    sklearn's `_compute_mi_cd`.
  - `InfoNCEEstimator` — bilinear critic `f(y, t) = exp(W[y]·t)`,
    cross-entropy on the BxB score matrix, lower bound `log(B) − L_NCE`.
  - `MINEEstimator` — 2x256 MLP critic, Donsker-Varadhan with MINE-f
    EMA bias-corrected gradient (Belghazi 2018 App C).

- `src_experiment/baselines/gen_gap_predictors.py` — four predictors,
  each returning a dict with the predictor value and a `wall` field:
  - `frobenius` — Σ_l ||W_l||_F.
  - `path_norm` — sqrt(Σ_paths Π_l w_e²) via the squared-weight forward
    trick. Also reports `log_path_norm`.
  - `spectral_margin` — γ_min(X, Y) and the spectral-product
    denominator Π_l ||W_l||_2; ratio is the predictor.
  - `sharpness` — top-`k` Hessian eigenvalues of the CE loss at θ via
    `scipy.sparse.linalg.eigsh` on a torch-autograd HVP. Subsamples
    to 1024 points by default; falls back to power iteration on
    Lanczos non-convergence.
  - `load_neural_net_from_h5(h5, epoch)` — rebuilds the
    `src_experiment.utils.NeuralNet` so the same PyTorch graph used at
    training time backs the HVP.

### Scripts

- `scripts/validate_mi_baselines.py` — synthetic-truth gate (planning
  §A.1.4). Closed-form discrete MI (binning/kmeans), MC-truth mixed
  Gaussian-discrete (KSG, six d×μ settings), MC-truth Gaussian-sign
  (InfoNCE/MINE). Exits 0/1.
- `scripts/run_mi_baselines.py` — per-HDF5 MI sweep. Reuses
  `run_label_noise_estimator.discover_jobs`. Extracts last-epoch
  deepest-layer pre-activation, subsamples to N=5000, runs all five
  baselines, emits `mi_baselines_seed_<seed>.csv` next to each HDF5.
  Resumable; `--aggregate` → `results/mi_baselines.csv`.
- `scripts/validate_gen_gap_predictors.py` — closed-form gate
  (planning §A.2.4). Frobenius on identity weights, path-norm on a
  2-layer all-ones model (closed form), spectral_prod on identity,
  sharpness against brute-force full Hessian on a tiny network.
  Exits 0/1.
- `scripts/run_gen_gap_predictors.py` — per-HDF5 gen-gap sweep.
  Mirrors `run_mi_baselines.py`. Emits `gen_gap_seed_<seed>.csv`
  next to each HDF5; `--aggregate` →
  `results/gen_gap_predictors.csv`. Also pulls
  `final_train_loss/test_loss/accuracy` and `gen_gap_loss/acc` from
  the HDF5 `training_results` group for direct correlation analysis.
- `run_phase_a_baselines.sh` — six-step top-level driver (A.1
  validate / sweep / aggregate, then A.2 validate / sweep /
  aggregate). Env switches: `PHASE_A_NO_VALIDATE=1` skips both
  gates, `PHASE_A_SKIP_MI=1` / `PHASE_A_SKIP_GG=1` skip a half. CLI
  args after the script name pass through to the sweep runners.

### Planning doc updates

- `planning/phase_a_baselines.md` extended with KSG (per request):
  new method subsection, `ksg_mi` API, mixed-Gaussian validation
  case, +45 min compute, curse-of-dimensionality risk row.

## Validation results (2026-04-28)

`validate_mi_baselines.py` (12 assertions) and
`validate_gen_gap_predictors.py` (7 assertions) both passed.

| baseline | check | Δ vs truth | tolerance |
|---|---|---|---|
| binning | discrete K=4 ε=0.2, N=10k | 0.010 bits | ±0.05 |
| kmeans | discrete K=4 ε=0.2, N=10k | 0.010 bits | ±0.05 |
| KSG | mixed Gaussian, d∈{1,5}, μ∈{0.5,1,2} | max 0.007 bits | ±0.05 (slack ±0.10) |
| InfoNCE | Gaussian-sign, ρ∈{0.5, 0.8} | max 0.021 bits | ±0.20 |
| MINE-f | Gaussian-sign, ρ∈{0.5, 0.8}, 3 seeds | max 0.012 bits, σ≤0.003 | ±0.10 |
| frobenius | identity weights, two layers, d=5 | 0 | exact |
| path_norm | 2-layer all-ones, in=2 hid=3 out=2 | 0 | exact |
| spectral_prod | identity weights | 0 | exact |
| gamma_min | identity logits, one-hot inputs | 0 | exact |
| sharpness λ_max | brute-force Hessian on 27-param net | <1e-3 | <1e-3 |
| sharpness Σ top-5 | brute-force Hessian | <1e-2 | <1e-2 |

## Smoke runs (single HDF5)

| HDF5 | what ran | wall | notes |
|---|---|---|---|
| `composite/n0.0_[25,25,25,25,25]/seed_101` (MI) | binning + kmeans + KSG | 6.6 s | k-means dominates (sweep × `n_init=10`) |
| `wbc/n0.0_[10,10,10,10,10]/seed_101` (MI) | full incl. MINE×3, InfoNCE×3 | 42.6 s | MINE 33 s, InfoNCE 8 s, rest <1 s |
| `composite/n0.0_[25,25,25,25,25]/seed_101` (gen-gap) | all 4 predictors, sharpness ×3 seeds | 0.8 s | sharpness 0.87 s, rest <10 ms; smaller than the plan estimate (~30 s) because models are small (1k–5k params) |

Sample MI headline numbers from the WBC smoke (binary class,
H(Y)≈0.95): ours_raw=0.658, kmeans-K|Y|=0.807, binning_4=0.834,
InfoNCE=0.848, KSG_k3=0.862, MINE-f=0.879. Order matches the planning
intuition (continuous estimators ≥ ours_raw > coarse plug-in
baselines).

Sample gen-gap row from the same composite HDF5: gen_gap_acc=0.001,
frobenius=18.76, log_path_norm=4.34, spectral_margin_ratio=−0.004,
λ_max(median over 3 seeds)=45.6, IQR=5.9, Lanczos converged.

## How to run later

```bash
# full pipeline: validate → sweep → aggregate, both halves
# (A.1 ≈ 6 h on CPU per the plan, A.2 ≈ 1 h since models are small):
./run_phase_a_baselines.sh

# A.1 only (skip gen-gap):
PHASE_A_SKIP_GG=1 ./run_phase_a_baselines.sh

# A.2 only (skip MI baselines):
PHASE_A_SKIP_MI=1 ./run_phase_a_baselines.sh

# A.1 fast tier (binning+kmeans+KSG only, no MINE/InfoNCE):
PHASE_A_SKIP_GG=1 ./run_phase_a_baselines.sh --skip-mine --skip-infonce

# pass-through filtering on the sweep runners:
./run_phase_a_baselines.sh --datasets wbc --noise 0.0 0.4
```

Aggregated outputs land at `results/mi_baselines.csv` and
`results/gen_gap_predictors.csv`.

## Plots & §5 writeup (done 2026-04-28)

- `scripts/plot_mi_baselines.py` (A.1.6) →
  `figures/baseline_mi_comparison.png`,
  `figures/baseline_mi_panel_b.png`,
  `figures/baseline_mi_vs_noise.png`, plus
  `results/baseline_mi_summary.csv`.
- `scripts/plot_gen_gap_predictors.py` (A.2.6) →
  `figures/baseline_gen_gap_kendall.png`,
  `results/gen_gap_predictors_kendall.csv`. Joins `ρ_func`,
  `rl_proxy`, and `Ĩ_ours` from
  `*_label_noise_new_estimator.csv` at ε = 10 (deepest layer × last
  epoch) — the sweep CSV's `epsilon_for_ours_func = 0.0001` is the
  degenerate "no merging" setting and is not the right ε for ranking
  against gen-gap.
- §5 written in `logging/paper_experiments_summary.md`. Headlines:
  KSG dominates the baseline cost-accuracy frontier (≈ 0.07 bits
  from MINE at ~0.06 s); `ρ_func` is the top cross-cell gen-gap
  predictor on composite (Kendall τ = +0.57, beats sharpness/
  path-norm/Frobenius), mid-pack on wbc (τ = +0.11) where sharpness
  λ_max wins (τ = +0.43); `rl_proxy` and `Ĩ_raw` flip sign across
  datasets and are now positioned as per-regime diagnostics rather
  than universal Jiang-style predictors.

## Not yet done

- Optional: revisit A.2's data choice — the predictors currently use
  the probe set rather than replaying the noise-injected training
  set. Documented in §5.0 setup; if a referee asks, swap to training
  data via `process_and_split` per (noise, arch, seed).
- Phase A on mnist: deferred until the noise-injected mnist runs
  from Open follow-up #2 in `paper_experiments_summary.md` land.
