# MNIST capacity sweep — quotient-MI experiment plan

This file lays out a focused experiment running the routing /
quotient-MI / RTG estimator over a *selection* of the trained
`outputs/mnist_capacity/` models. Picks up on the headline composite
finding — quotient MI is more sensitive to label noise on small / medium
nets, but flips on wide nets that memorize — and tests whether the
**capacity-vs-bottleneck** axis on real data shows the same separation
without label noise (using PCA dim as the information bottleneck instead).

## What's available in `outputs/mnist_capacity/`

- 49 (PCA dim × arch) cells × 5 seeds = **245 trained models**.
- PCA dims: `{2, 3, 4, 5, 10, 15, 20}` (`target_dim` in metadata).
- Architectures (depth = 3): `{[3]^3, [5]^3, [7]^3, [15]^3, [25]^3, [50]^3, [100]^3}`.
- Each HDF5 carries 22 epochs of saved tree state + the **full MNIST
  test set (N = 10 000)** at the PCA-reduced, MinMax-scaled feature
  space. Class distribution is natural MNIST, ≈ balanced; H(Y) ≈ 3.32 bits.

The stored test points are already in the model's training-time feature
space, so the estimator can use them as-is with no new probe loader.

## Hypotheses

H1 — **Information-bottleneck × capacity creates a clean memorization /
generalization landscape** that quotient MI separates better than raw
plug-in MI. Specifically: at high capacity (wide nets) and tight
bottleneck (low PCA dim), the network has more "room to memorize"
relative to the available task information; raw MI may *over*-report
this, quotient MI should attribute less of it to genuine class
information.

H2 — **`I_func / I_raw` (the quotient-to-raw ratio) varies systematically
across the (PCA dim, width) grid.** The ratio should approach 1 in the
"clean generalization" cells (enough info, small enough net) and drop
when the network has surplus capacity carving structurally-distinct but
functionally-redundant regions.

H3 — **Trustworthiness ρ scales predictably**: increases with width and
PCA dim. We expect a sub-region of the grid where ρ < 0.3 (trustworthy)
and a wider region where the qualitative ordering still holds.

## Selection (Phase A — first pass)

Goal: map the (capacity × bottleneck) landscape with the fewest jobs that
still let us answer H1–H3.

Grid:

| axis | values | rationale |
|---|---|---|
| PCA dim | `{2, 5, 10, 20}` | extremes plus two interior points along the bottleneck axis |
| arch | `{[5]^3, [15]^3, [25]^3, [100]^3}` | narrow → wide, bracketing the composite memorization threshold (~ [25]) |
| seeds | `{101, 102, 103}` | three for noise on the headline numbers; seeds 104/105 reserved for densification if Phase A is interesting |

**Total Phase A: 4 × 4 × 3 = 48 jobs.**

Conservative wall-time estimate: composite at N=20 000 averaged
15.8 s/job; MNIST at N=10 000 with depth 3 (vs composite up to depth 5)
should be faster — call it ~10 s/job → **~8 min** for Phase A.

ε grid: same as composite/wbc — `(0, 1e-4, 1e-2, 1e-1, 1, 10, 100, 1000)`.
The ε = 10 inflection regime is expected to transfer (it has on both
prior datasets).

## Phase B — defer

If Phase A reveals a clean H1/H2 pattern, Phase B will (a) add the
remaining seeds to nail down 5-seed CIs and (b) fill in the missing
`{3, 4, 15}` PCA dims and `{3, 7, 50}` archs along the boundary that
matters. Concretely: if memorization shows up around `[25]^3` × PCA=2,
densify around that cell.

If Phase A is *uninformative* (no clean separation), revisit before
spending more compute.

## Pipeline

1. **New driver: `run_mnist_capacity_estimator.py`.** Walks the selected
   `(PCA dim, arch, seed)` cells, calls
   `FunctionalQuotientEstimator(h5).evaluate_all(...)` (no probe — uses
   stored test data), tags rows with `dataset='mnist'`,
   `target_dim=<pca>`, `arch_str=<arch>`. Writes
   `outputs/mnist_capacity/<pca>_dim_<arch>/new_estimator_seed_<seed>.csv`.
   Resumable (skips existing CSVs).
2. **Aggregate**: same script with `--aggregate`, output to
   `results/mnist_capacity_new_estimator.csv`.
3. **Sanity checks** mirror `scripts/composite_sanity_check.py`:
   plug-in DPI, MM DPI delta, max ρ per (pca, arch), `truncation_prob`
   (will be NaN — no holdout), plateau location.
4. **Headline plots**:
   - **Landscape grid**: `I_raw` and `I_func` heatmaps over (PCA dim,
     arch) at the deepest layer, ε = 10, last epoch. Two side-by-side
     heatmaps — quickest readout of H1.
   - **Quotient-ratio heatmap**: `I_func / I_raw` over the same grid —
     direct visualization of H2.
   - **Per-cell training curves** for 4 representative cells (clean
     generalization, memorization, bottlenecked, capacity-rich) at
     ε = 10.
5. **Brief log**: `logging/mnist_capacity_phaseA_summary.md` mirroring
   the WBC / composite summary structure.

## Where reused code lives

- `src_experiment/functional_quotient.py:FunctionalQuotientEstimator` —
  HDF5-driven driver, already supports default-probe path.
- `run_label_noise_estimator.py` — pattern for `discover_jobs`,
  `run_jobs`, `aggregate`. We mirror its structure but key by
  `(target_dim, arch, seed)` instead of `(noise, arch, seed)`.
- `visualization/plot_new_estimator.py` — already accepts arbitrary
  per-row CSVs; the noise-compare flag won't apply (no noise axis), but
  the dashboard mode does. Heatmaps will be a small new plotter.

## Out of scope (this phase)

- The `mnist_minimal_random` (random-label) sweep in
  `outputs/mnist_mem_gen_exp/`. Worth picking up after if Phase A
  shows H1, since it's a clean memorization benchmark.
- Label noise on MNIST. The training pipeline supports it but no such
  sweep has been trained for `mnist_capacity`.
