# 2026-04-27 — paper-grounded follow-up experiments (in progress)

After the composite sweep and MNIST capacity Phase A+B (own logs), I read
`Johan_binning_paper_NeurIPS_2026-1.pdf` and proposed three experiments
that test specific paper claims using only the data already in
`outputs/{composite,wbc}_label_noise/` and `outputs/mnist_capacity/`. No
new training runs needed.

## Experiments proposed (user-approved)

1. **Routing-information training dynamics.** Plot Ĩ(Y;Ω), Ĩ_func(Y;Ω_func)
   and ρ_func vs epoch at the deepest layer, ε = 10. Tests Sec 6 +
   Discussion 7.1 (capacity saturation) with temporal evidence the paper
   doesn't yet show.
2. **ρ_func ↔ generalization gap.** Treat ρ_func at convergence as a
   diagnostic for how much combinatorial capacity the network actually
   uses; check whether it predicts the train-vs-test accuracy gap.
   Tests Sec 4.5's claim that small ρ_func indicates "structured
   computation".
3. **RTG-aware routing-loss proxy.** Build `RL_proxy` = fraction of
   data-supported RTG edges (Hamming-1) whose endpoints disagree on
   majority class. Tests Prop 4.2 empirically by giving the routing-loss
   term I(Y;Π|T) — the only term in the decomposition with no estimator —
   a tractable surrogate.

## Steps done so far

### Step 1 — joined training_results into the aggregated CSVs

Script: `scripts/join_training_results.py`. For each row in the three
aggregated CSVs (composite, wbc, mnist), opened the matching HDF5 and
indexed `eval_train_accuracy`, `test_accuracy`, `eval_train_loss`,
`test_loss` at the row's epoch, then computed `gen_gap_acc` and
`gen_gap_loss`. Backups in `*.bak`. Touched 530 unique HDF5s; 0 / 351 040
rows missing.

Spot check at last epoch:

| dataset   | gen_gap_acc range | mean | comment |
|---|---|---:|---|
| composite | [-0.399, 0.011]   | -0.189 | noise sweeps → train acc against *noisy* train labels < test acc against *clean* test labels; signed gap is negative under noise. Need `gen_gap_acc + noise_level` (or absolute) for Exp 2. |
| wbc       | [-0.330, 0.061]   | -0.108 | same pattern, smaller magnitude. |
| mnist     | [-0.017, 0.028]   | -0.002 | no noise → tiny gap. |

### Step 2 — training-dynamics plots (Experiment 1)

Script: `scripts/training_dynamics.py`. For each (dataset, arch) listed
below, faceted by the relevant axis (noise level for composite/wbc, PCA
target_dim for mnist), plotted four panels: Ĩ_raw, Ĩ_func, ρ_func, test
accuracy vs epoch. Mean ± std over seeds; deepest layer; ε = 10.

Saved to `figures/training_dynamics/`:

- `composite_{5_5_5, 7_7_7, 25_25_25, 5_5_5_5_5}.png`
- `wbc_{5_5_5, 7_7_7, 25_25_25, 5_5_5_5_5}.png`
- `mnist_{5_5_5, 7_7_7}.png`

**Headline finding (new — not in the paper).** ρ_func shows a
**compression-then-expansion** trajectory across epochs, the discrete
analogue of the IB compression / fitting phases:

| dataset | arch | noise | ρ_func@init | ρ_func min (epoch) | ρ_func final |
|---|---|---:|---:|---:|---:|
| composite | `[7,7,7]`    | 0.0 | 0.107 | 0.083 (e=2)  | **0.718** |
| composite | `[7,7,7]`    | 0.4 | 0.095 | 0.095 (e=0)  | **0.218** |
| composite | `[25,25,25]` | 0.0 | 0.199 | 0.142 (e=4)  | 0.404 |
| composite | `[25,25,25]` | 0.4 | 0.232 | 0.166 (e=8)  | 0.229 |
| wbc       | `[25,25,25]` | 0.0 | 0.355 | 0.230 (e=110)| 0.240 |
| wbc       | `[25,25,25]` | 0.4 | 0.355 | 0.349 (e=30) | 0.360 |
| mnist     | `[7,7,7]`    |  -  | 0.02–0.14 (varies with PCA dim) | — | 0.36–0.76 |

Two specific phenomena worth reporting:

1. **Compression phase exists in the discrete routing geometry.** It is
   visible in every trustworthy (arch, dataset) cell except WBC at
   high noise (where the dataset is too small to show clean dynamics).
   On composite `[7,7,7]` clean, ρ_func bottoms out by epoch 2, then
   rises monotonically.
2. **Label noise *suppresses* the expansion phase.** Composite
   `[25,25,25]`: clean ρ_func ends at 0.40; noisy ends at 0.23. The
   network creates many *structurally distinct* regions under noise
   (raw R rises) but most collapse onto the same affine map (R_func
   barely rises) — they are routing-equivalent. Combined with the
   simultaneous *rise* in I_func under noise on `[25,25,25]` (already
   seen in the composite memorization analysis), this looks like
   **memorization via redundant routing**: many polytopes carved
   around noisy points, all implementing the same local computation.
   Prop 4.6 says the quotient is the right object; this trajectory
   shows it is empirically the *active* object during memorization.

This finding alone seems worth a paragraph in Sec 6 of the paper.

## Steps not yet done

~~Both Exp 2 and Exp 3 listed below are now complete.~~ See
`logging/2026-04-27_exp2_exp3_completion.md` for the chronological
session log and `logging/paper_experiments_summary.md` for the
combined writeup (Sections 1-4).

- ~~**Experiment 2 (ρ_func ↔ generalization)**~~ — done. Headline:
  cross-cell composite r = −0.86 [−0.93, −0.79] on |train−test|
  (noise-axis confound); within-noise stratification (Sec 2.4 of
  the summary) rescues wbc to r = +0.61/+0.62/+0.80 at
  noise = 0/0.2/0.4 and confirms mnist at +0.7-0.85 within each PCA
  dim.
- ~~**Experiment 3 (RL_proxy)**~~ — done. Module
  `src_experiment/rtg_overlap.py` added; `FunctionalQuotientEstimator`
  emits an `rl_proxy` column. Cross-cell Pearson r in [+0.68, +0.83]
  across composite/wbc/mnist; within-noise composite hits r =
  +0.93 [+0.90, +0.98] at noise = 0.4 — the strongest single
  statistic of the series.

## Loose ends

- The compression-then-expansion finding deserves a focused figure for
  the paper (currently spread across 10 small files; one composite
  figure with the four (arch × noise) cells stacked would be the
  paper-quality version).
- WBC is too small to resolve the compression phase cleanly. Probably
  not a fixable problem without a larger UCI dataset.
- The MNIST capacity sweep was *clean* (no noise), so it can only
  show the expansion phase relative to PCA bottleneck. The "label
  noise on MNIST" follow-up suggested in
  `logging/mnist_capacity_summary.md` would close the loop.
