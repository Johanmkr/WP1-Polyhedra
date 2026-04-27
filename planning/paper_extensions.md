# Paper extensions — plan toward NeurIPS main-track readiness

The current paper-grounded experiment series (Exp 1-3, see
`logging/paper_experiments_summary.md`) gives clean validation of the
new estimator on three datasets, but the empirical scope is too
narrow for top-tier general venues — see review-style assessment in
the conversation log preceding this plan.

This file plans the three extensions ordered by *review-impact per
unit compute*: **Phase A (baselines)** is the highest-leverage
addition; **Phase B (larger tabular UCI)** is the cheapest; **Phase
C (full MNIST + small CNN)** is the most code-heavy and gives the
biggest scope extension. Total calendar estimate: 6–9 weeks. They
can be parallelised where independent compute / dev resources allow.

---

## Phase A — Baseline comparisons (~2-3 weeks)

The single highest-value addition. The current draft has no
head-to-head against existing methods; reviewers cite this almost
unanimously as a reject-trigger. Two distinct comparisons map onto
two distinct claims in the paper.

### A.1 — MI baselines for `Ĩ(Y;Ω)` (Recipe 1)

**Compare against:**
1. **MINE** (Belghazi et al., 2018) — neural variational MI lower
   bound. Trains a discriminator network on (Y, T) pairs vs marginals.
2. **InfoNCE** (van den Oord et al., 2018) — variational MI lower
   bound via contrastive estimation. Cheaper than MINE.
3. **Plug-in MI on raw activations**, post-quantization (the
   Tishby-style binning baseline). Bin pre-activations into ~8 bins
   per neuron, hash, plug-in.
4. **Discrete-symbol MI on K-means clusters of activations** (cluster
   T into K groups, plug-in MI on the cluster IDs). K = number of
   ground-truth classes is the natural choice; sweep K = {2, 5, 10,
   50}.

**Where:** all three trustworthy datasets (composite, wbc, mnist
trustworthy subset), at the deepest layer, last epoch. Network ID
= 555 evaluator runs already on disk; the comparators evaluate
the same `(W, b, X, y)` and produce a single MI scalar per row.

**Implementation:**
- `src_experiment/baselines/mi_baselines.py`:
  - `mine(activations, labels, n_iters=2000)` — MLP discriminator
    + Donsker-Varadhan estimator. Reuse the
    `Anthropic/torch-mine`-style implementation pattern; ~150 LoC.
  - `infonce(activations, labels, n_iters=2000)` — contrastive
    variant; ~100 LoC.
  - `binning_mi(activations, labels, n_bins=8)` — quantize then plug
    in. ~30 LoC.
  - `kmeans_mi(activations, labels, K)` — cluster then plug in.
    ~30 LoC.
- `scripts/run_mi_baselines.py` — walks the same HDF5 set as the
  estimator runners, evaluates each baseline per (epoch, layer),
  writes a per-HDF5 CSV with columns
  `{method, mi_estimate, wall_seconds}`.
- `scripts/aggregate_mi_baselines.py` — concatenates into
  `results/mi_baselines.csv`.

**Dev cost:** ~3-4 days. Each baseline is a well-known recipe.

**Compute cost:**
- MINE: ~5-10 min per (epoch, layer) × 22 epochs × 3-5 layers per
  HDF5 × 555 HDF5s ≈ **~3-5 days** on GPU. Will dominate.
  *Can be reduced* by evaluating only at last epoch + deepest
  layer (matches the headline plot in the paper) → ~2 hours.
- InfoNCE: ~2× faster than MINE.
- binning + K-means: minutes per HDF5 total.

**Recommendation:** evaluate baselines only at (last epoch, deepest
layer) to match the headline plot. Full-grid MINE is overkill for
the comparison.

**Output table:**

| dataset | method | I-estimate (mean ± std) | wall/HDF5 | Pearson r vs gen-gap |
|---|---|---|---|---|
| composite | Ĩ_raw (ours) | … | <1 s | … |
| composite | Ĩ_func (ours) | … | <1 s | … |
| composite | MINE | … | ~5 min | … |
| composite | InfoNCE | … | ~2 min | … |
| composite | binning (8 bins) | … | <1 s | … |
| composite | K-means (K=Y) | … | <10 s | … |
| (× wbc, mnist) | … | … | … | … |

The pitch: ours matches MINE within DPI consistency at fraction of
the cost; the binning/k-means baselines miss either DPI (binning
explodes) or class-information (k-means with K = |Y| is a lossy
quotient).

### A.2 — Generalization-gap predictor comparison

**Compare ρ_func at convergence against:**
1. **Sharpness** (Keskar et al., 2017) — top-k Hessian eigenvalues
   approximated by Lanczos / power iteration.
2. **Spectrally-normalized margin** (Bartlett et al., 2017; Jiang
   et al., 2020 simplified form).
3. **Path-norm** (Neyshabur et al., 2015).
4. **Frobenius norm of weights** (the trivial baseline).
5. **Final-epoch test loss** (also trivial; the real-train baseline).

**Where:** the same 555 HDF5s, deepest layer, last epoch. Each
predictor is a single scalar per HDF5; we then compute Pearson r
between predictor and gen-gap (using the within-noise stratification
from Sec 2.4 to avoid noise-axis confounds).

**Frame as a `Jiang et al. 2020`-style generalization-measure
benchmark.** That paper laid down the protocol and gives a strong
referee target; even mentioning it pre-empts the most common
"how is this measured" objection.

**Implementation:**
- `src_experiment/baselines/gen_gap_predictors.py`:
  - `sharpness(model, dataloader, k=5)` — Lanczos via
    `scipy.sparse.linalg.eigsh` on the Hessian-vector product.
    ~80 LoC.
  - `spectral_margin(model, X, y)` — operator norms of weight
    matrices × min train-set margin. ~50 LoC.
  - `path_norm(model)` — recursive sum-product over layers.
    ~30 LoC.
  - `frobenius(model)` — trivial.
- `scripts/run_gen_gap_predictors.py` — emits one row per HDF5.
- Compares Pearson and Kendall τ (Jiang et al.'s preferred
  rank-correlation for generalization predictors).

**Dev cost:** ~3-4 days. Sharpness via eigsh is the only non-trivial
piece.

**Compute cost:** ~30 s per HDF5 × 555 ≈ **~5 hours**. Cheap.

**Output:** scatter of each predictor vs gen-gap (cross-cell and
within-noise), Kendall τ table à la Jiang et al. Decide on the
basis of *replication across datasets*: if ρ_func ranks #1 or #2
on the Kendall τ table on at least two of three datasets, the paper
can claim "competitive with or better than X for predicting
generalization."

### A.3 — Risk

- MINE is finicky to train; reviewers may attack the comparison if
  the MINE numbers look anomalously low. Mitigate by reporting the
  multi-seed mean ± std and documenting hyperparameter choices.
- The sharpness baseline is famously variance-prone; report median +
  IQR across 5 evaluation seeds.

### A.4 — Deliverables

- `src_experiment/baselines/{mi_baselines,gen_gap_predictors}.py`
- `scripts/run_mi_baselines.py`,
  `scripts/run_gen_gap_predictors.py`
- `results/mi_baselines.csv`, `results/gen_gap_predictors.csv`
- `figures/baseline_mi_comparison.png`,
  `figures/baseline_gen_gap_kendall.png`
- New section in `logging/paper_experiments_summary.md` —
  "Section 5: Baseline comparisons".

---

## Phase B — Larger tabular UCI (~1-2 weeks)

WBC at N = 569 carries too much weight in the cross-task
replication argument (Sec 4 of the summary). A mid-size tabular
UCI dataset gives a third anchor point with tighter CIs.

### B.1 — Dataset choice

Top candidate: **Forest Cover Type (UCI ID 31)**.

- N = 581 012, 7 classes, 54 numeric features (including 44 binary
  one-hot for soil/wilderness types).
- Numeric features → MLP-friendly with no preprocessing beyond
  `MinMaxScaler` to `[-1, 1]`.
- 7 classes give H(Y) ≈ 1.5 bits — large enough for the gen-gap
  signal not to saturate (unlike binary tasks).
- Already supported by `ucimlrepo`.

**Alternatives considered and rejected:**
- Adult (N = 48k, binary income) — binary, low H(Y).
- Bank Marketing (N = 45k, binary) — same.
- HEPMASS / HIGGS (N = 10M+) — too big to train at sweep scale.
- Letter Recognition (N = 20k, 26 classes) — interesting H(Y) but
  small N relative to covertype.

### B.2 — Subsample to N = 100k

Full N = 581k is overkill for the estimator and would make training
slow at sweep scale. Stratified subsample to N = 100k (15 % of
total) with a fixed seed — leaves N_train ≈ 80k, N_test ≈ 20k,
which is plenty for ρ-trustworthiness on archs up to `[25,25,25]`.

### B.3 — Sweep grid

Mirror the wbc/composite sweep so Sec 4 can replicate cleanly:

- archs: `{[5,5,5], [7,7,7], [9,9,9], [25,25,25], [5,5,5,5,5],
  [7,7,7,7,7]}` (6 archs)
- noise: `{0.0, 0.2, 0.4}`
- seeds: 5 (101..105)
- Total: 6 × 3 × 5 = **90 trained networks**.

### B.4 — Implementation

- Add `make_covertype_split(noise_level, target_dim=None, ...)` to
  `src_experiment/dataset.py` mirroring the wbc loader. Accept the
  N=100k stratified-subsample seed as a parameter.
- Add a `covertype_label_noise` config in `configs/` mirroring
  `configs/wbc_label_noise.yaml`.
- Add `make_covertype_probe(global_seed=42, mode='full', ...)` to
  `src_experiment/probe_loader.py`.
- Extend `run_label_noise_estimator.py`'s `DATASETS` dict to include
  covertype.
- Run the chain.
- Re-aggregate, run the existing dynamics + Exp 1-3 plotting scripts
  with covertype added.

### B.5 — Compute cost

- Training (per network): ~5-10 min for N_train = 80k × 150 epochs
  × MLP on CPU; ~2-5 min if GPU available. 90 nets × 8 min ≈
  **~12 hours**.
- Estimator: per-HDF5 wall on the existing wbc sweep was ~4 s/job
  for `[7,7,7]`-class archs; covertype's larger probe will scale
  linearly in N. Estimate ~30-60 s/job × 90 = **~1 hour**.
- Plot regeneration: minutes.

### B.6 — Deliverables

- `src_experiment/dataset.py` — covertype loader.
- `src_experiment/probe_loader.py` — covertype probe builder.
- `configs/covertype_label_noise.yaml`.
- `outputs/covertype_label_noise/` — trained HDF5s.
- `results/covertype_label_noise_new_estimator.csv`.
- `figures/training_dynamics/covertype_*.png`.
- Section 4 of `paper_experiments_summary.md` — extended replication
  table now includes covertype.

### B.7 — Risk

- Covertype is famously imbalanced (class 1 ≈ 36 %, class 4 ≈ 0.5 %).
  Report per-class precision/recall, and consider class-balanced
  loss for one ablation seed if the imbalance dominates.
- N = 80k may push the Recipe 2 active-subnetwork clustering past
  the 10k-region threshold for the wider archs (`[25,25,25]` at
  PCA = full). If so, fall back to the LSH variant or restrict
  Phase B to the narrow archs only.

---

## Phase C — Full MNIST + small CNN (~3-4 weeks)

The hardest extension and the most paper-impactful: full MNIST
(no PCA) on a small CNN. This closes both the "doesn't scale to
real data" and "doesn't scale beyond MLPs" critiques.

### C.1 — Architecture choice

**LeNet-5 family**, parametric in width. Default:

```
Input: 1 × 28 × 28
Conv1: 6 channels, 5×5, stride 1, ReLU            -> 6 × 24 × 24
Pool1: 2×2 max                                    -> 6 × 12 × 12
Conv2: 16 channels, 5×5, stride 1, ReLU           -> 16 × 8 × 8
Pool2: 2×2 max                                    -> 16 × 4 × 4
Flatten                                           -> 256
FC1: 120, ReLU                                    -> 120
FC2: 84, ReLU                                     -> 84
FC3: 10                                           -> 10 (logits)
```

Sweep narrowly on width:
- LeNet-XS (4, 8, 60, 42)
- LeNet-S (6, 16, 120, 84) — default
- LeNet-M (8, 24, 180, 126)
- LeNet-L (12, 32, 240, 168)

### C.2 — Estimator extension to ReLU CNNs

Conv-ReLU layers fit the Recipe 1-3 framework cleanly because each
ReLU is still elementwise gating: a region is the cumulative pattern
of which (channel, spatial-position) ReLUs are active across all
layers.

**Two design decisions:**

1. **Cumulative-pattern length.** For LeNet-S the per-image binary
   pattern is `6·24·24 + 16·8·8 + 120 + 84 = 3456 + 1024 + 120 + 84
   = 4684 bits`. Hashing this is fine
   (`np.packbits` + `md5.digest` already handles arbitrary length).
   The Hamming-1 RTG enumeration cost grows linearly in this length
   per region: ~5k single-bit flips per region per query.
   At 10k regions this is 50M dict lookups — manageable but slow.
   *May need:* the LSH variant for Recipe 2 if region count
   saturates above ~30k.

2. **Active-subnetwork matrix for conv layers.** The recursion
   `Ã_i = W^i[S_i, S_{i-1}] · Ã_{i-1}` extends to convs by
   *unrolling* the conv as a sparse linear operator: for a 5×5 conv
   on H×W input with K output channels, the unrolled weight is a
   sparse `(K·H'·W') × (C·H·W)` matrix. The active output set
   `S_l` is the subset of (channel, spatial-position) pairs with
   ReLU on. Pool layers are deterministic max gates and can be
   absorbed by tracking which input position fired.

   *Implementation:* either (a) materialize the sparse unrolled
   conv as `scipy.sparse.csr_matrix` (memory-feasible at LeNet
   scale: ~1M nonzeros for conv1) and reuse the existing FC code,
   or (b) implement a conv-aware `compute_active_subnetwork` path.
   Option (a) is faster to write and debug.

**Dev work:**
- `src_experiment/utils.py` — `LeNet5(nn.Module)` parametric on
  channel widths. ~80 LoC.
- `src_experiment/cnn_estimator.py` (new) — `forward_activation_patterns_cnn`
  that returns per-layer binary masks (including conv layers).
  Wraps existing `routing_estimator` machinery via the unrolled
  sparse weight trick.
- `src_experiment/functional_quotient.py` — extended
  `compute_active_subnetwork` to dispatch on layer type (conv/FC).
- Smoke test: confirm `Ĩ_raw` and `Ĩ_func` recover the FC-only
  numbers when the CNN is configured with no conv layers. Validates
  the implementation against the existing pipeline.

**Dev cost:** **~2 weeks**. The unrolled-conv trick is the central
risk; budget a week for it.

### C.3 — Sweep grid

- archs: 4 LeNet widths (XS, S, M, L)
- noise: `{0.0, 0.2, 0.4}` — same noise loader as composite, applied
  to MNIST training labels
- seeds: 5 (101..105)
- Total: 4 × 3 × 5 = **60 trained networks**.

Probe: full MNIST test set (N = 10 000), no PCA. Holdout: regenerate
from the train split (N = 5 000 disjoint).

### C.4 — Compute cost

- **Training:** small CNN on MNIST trains in ~10-30 min per run on
  CPU with N_train = 60k × 150 epochs (with `save_interval=10`),
  ~3-5 min on a single GPU. 60 runs × 15 min ≈ **~15 hours**.
- **Estimator:** dominated by the active-subnetwork unroll. With N
  = 10 000 probe and pattern length ~5000 bits per (epoch, layer),
  expect ~5-15 min per HDF5 (depends on R). 60 × 10 min ≈
  **~10 hours**.
- **Total:** ~1 day of compute, mostly on a single machine.

If the LSH variant is needed, the estimator side may double
(~20 hours).

### C.5 — Deliverables

- `src_experiment/utils.py` — LeNet5 class.
- `src_experiment/cnn_estimator.py` (new).
- `src_experiment/functional_quotient.py` — conv-aware path.
- `configs/mnist_full_lenet.yaml`.
- `outputs/mnist_full_lenet/` — trained HDF5s.
- `results/mnist_full_lenet_new_estimator.csv`.
- `figures/training_dynamics/mnist_full_lenet_*.png`.
- `figures/rho_func_vs_gen_gap_full_mnist.png`,
  `figures/rl_proxy_vs_quotient_gap_full_mnist.png`.
- New section in `logging/paper_experiments_summary.md` —
  "Section 6: Full MNIST with LeNet-style CNN".

### C.6 — Risk

1. **Recipe 2 active-subnetwork unrolling for conv layers may be
   buggy.** The smoke-test checkpoint (compare against an FC-only
   "CNN" with 1×1 convs) is critical. Budget extra dev time.
2. **Region count may saturate.** With LeNet-S × N = 10k probe,
   unique-region count per (epoch, layer) might exceed 10k at deep
   layers, requiring LSH. Sample a single training run early in
   development to verify; if saturated, immediately budget the LSH
   work in this phase rather than as a follow-up.
3. **MNIST is "solved"** — clean MNIST gen-gap is ~1 %, hard to
   tell ρ_func diagnostics from noise. The noise sweep at
   `noise = 0.4` is the regime that gives a measurable signal.
   This is anticipated; design the headline plot around the
   noise-stratified result, not the clean one.

---

## Suggested ordering

If compute is the binding constraint:

```
week  1-2:  Phase A.1 (MI baselines) + Phase B implementation
week  3:    Phase A.2 (gen-gap predictors) + Phase B compute
week  4:    Phase A writeups; Phase C dev (CNN estimator)
week  5-6:  Phase C dev (active subnetwork unrolling), smoke tests
week  7:    Phase C compute
week  8:    Phase C writeup; combined Section 5+6 of summary
week  9:    Buffer + figure polish + paper rewrite
```

If dev time is the binding constraint, do Phase B first (cheapest;
covertype is a one-week add) while Phase A.1 baselines run on
existing HDF5s in parallel.

## Submission target reassessment after each phase

- **After Phase B**: the cross-task replication anchor is
  strengthened. TMLR / workshop-grade paper is comfortably ready.
- **After Phase A**: a baseline-comparison table closes the most
  common reject reason. Borderline NeurIPS-main-track ready, with
  the architecture-scope objection still open.
- **After Phase C**: full MNIST + a CNN closes the architecture
  objection. NeurIPS / ICML main-track submission becomes credible.
- **Still open after all three**: CIFAR-10 + ResNet-class network.
  Not blocking for first submission, but a likely Reviewer-2 ask.
  Plan it as a follow-up paper or a rebuttal-cycle add.

## Out-of-scope for this plan

- CIFAR-10 + ResNet — separate planning doc; needs the LSH variant
  of Recipe 2 to be production-ready first.
- Vision transformer extensions — not feasible without rethinking
  the ReLU-region framework.
- A regularization / early-stopping operational story — orthogonal
  add-on; would push the paper from "describes" to "predicts and
  intervenes." Plan separately if the Phase A.2 predictor result
  comes out competitive.
