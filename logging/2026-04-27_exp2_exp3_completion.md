# 2026-04-27 — Experiments 2 and 3 completed (paper-grounded series)

Successor to `2026-04-27_paper_grounded_experiments.md` (which logged
Step 1: training-results join, and Step 2: Experiment 1 dynamics
plots). This log covers the same calendar day's later session: the
follow-through on Experiments 2 and 3 plus Section 4 of the combined
writeup.

The combined topic-organized writeup lives at
`logging/paper_experiments_summary.md` (Sections 1-4 + open
follow-ups). This log is the chronological session record.

## Scope of this session

1. (a) Polish the Experiment 2 figure that came out of the first
   pass (legend overlap, saturated y-axis on composite).
2. (b) Within-noise / within-PCA robustness check for Experiment 2
   to control for the noise-axis confound.
3. (c) Implement Experiment 3 (`RL_proxy`), re-run the estimator on
   composite + wbc + trustworthy mnist, plot, and write up Section 3.
4. Section 4 — cross-dataset replication table.
5. Within-noise stratification for Experiment 3 (added after the
   table).

## Step (a) — Experiment 2 figure cleanup

`scripts/rho_func_vs_gen_gap.py` was rewritten to use:
- color = network depth (3 = blue, 4 = green, 5 = red), saturation
  modulated by width;
- marker shape = noise level / target_dim;
- two shared bottom legends per dataset column (arch + facet);
- visible zero line to make the composite saturation legible
  (gen_gap_norm spans only ~10⁻² there).

Output: `figures/rho_func_vs_gen_gap.png`.

## Step (b) — within-noise robustness check (Section 2.4)

Same slice (deepest layer, ε = 10, last epoch) but correlations
recomputed *within* each noise level / target_dim instead of across
all cells. Saved to `results/rho_func_vs_gen_gap_within.csv`.
Findings:

| dataset | facet | n | r [95% CI] |
|---|---|---:|---|
| composite | noise = 0.0 | 12 | −0.52 [−0.89, −0.09] (saturated, see Sec 2.4) |
| composite | noise = 0.2/0.4 | 12 | null |
| wbc | noise = 0.0 | 18 | **+0.61** [+0.24, +0.86] |
| wbc | noise = 0.2 | 18 | **+0.62** [+0.05, +0.87] |
| wbc | noise = 0.4 | 18 | **+0.80** [+0.52, +0.94] |
| mnist | per target_dim | 6 | +0.73 to +0.86 across all PCA dims |

The cross-cell wbc result that looked null (r ≈ +0.31) is rescued
by the stratification — within each noise regime the predicted
positive correlation is strong and consistent. Composite is
saturated at zero noise (gen_gap_norm ~ 0 because composite
networks regularize) and is the wrong test bed.

## Step (c) — Experiment 3

### Implementation

- New module `src_experiment/rtg_overlap.py` exposing
  `region_dominant_class(omega_ids, y)` and
  `routing_loss_proxy(adjacency, dominant)`. Returns the fraction of
  unique Hamming-1 RTG edges whose endpoints disagree on majority
  class. ε-independent.
- `src_experiment/functional_quotient.py` modified:
  - `QuotientResult` gained `rl_proxy: float`.
  - `evaluate_epoch` computes `region_dominant_class` from
    `(omega, y)` and calls `routing_loss_proxy` once per
    `(epoch, layer)`; value duplicated across ε rows.
  - `evaluate_all` added `rl_proxy` column to the output schema.

### Compute (`logs/exp3_chain_20260427_181714.log`)

A single `nohup` chain script ran three sweeps with `--force`:

| sweep | jobs | wall |
|---|---:|---:|
| composite (`--composite-probe-size 20000`) | 180 | 49.3 min |
| wbc (`--wbc-mode full`) | 270 | 17.0 min |
| mnist (`--archs '[3,3,3]' '[5,5,5]' '[7,7,7]'`) | 105 | 10.2 min |

All 555 jobs succeeded. Total wall ≈ 76 min.

### Two operational gotchas

1. **`run_label_noise_estimator.py --aggregate` collapses both
   composite and wbc per-HDF5 CSVs into the single `--output` path.**
   After the chain, `results/composite_label_noise_new_estimator.csv`
   actually contained both datasets. I split them by `dataset`
   column and overwrote both target files manually. The aggregator
   needs a dataset filter to make this clean; for now, see the
   memory note.
2. **Per-HDF5 CSVs do not include `train_acc`/`test_acc`/etc.**
   Those columns are joined post-hoc by
   `scripts/join_training_results.py`. The rerun blew them away;
   re-running the joiner restored them (530 HDF5s touched, 0 rows
   missing). Both quirks now in `project_routing_estimator.md`
   memory.

### Plot and findings

`scripts/rl_proxy_vs_quotient_gap.py` produces
`figures/rl_proxy_vs_quotient_gap.png`. Pearson r [95% CI] across
(arch, facet) cells, mean over seeds:

- composite: **+0.77** [+0.60, +0.87] (n = 36)
- wbc: **+0.71** [+0.58, +0.82] (n = 54); ρ ≤ 0.3 trust subset
  **+0.83** [+0.73, +0.94] (n = 25)
- mnist: **+0.68** [+0.37, +0.84] (n = 21)

All bootstrap CIs sit entirely above zero. This is the cleanest
empirical confirmation of Prop 4.2 in the data. Figure shows a
saturating sigmoid on composite (RL_proxy ≈ 0 → gap ≈ 0;
RL_proxy ≈ 0.5 → gap ≈ 0.4 bits).

## Section 4 — cross-dataset replication table

Six findings × three datasets, scored ✓ / ◐ / ∅ / —. Score totals:
mnist 4.5 / 5, composite 3.5 / 6, wbc 3.5 / 6. Reading: Exp 3 is the
single finding that replicates everywhere with the right sign and
confidence; Exp 2 cross-cell flips sign by dataset because of the
noise confound; Exp 1 dynamics is qualitatively reproduced
everywhere but magnitudes are dataset-dependent. Section 4 of
`paper_experiments_summary.md` has the full table and reading.

## Step 5 — within-noise stratification for Experiment 3 (Section 3.1)

After the table I added a within-noise robustness check for
Experiment 3 too (parallel to Section 2.4). File:
`results/rl_proxy_vs_quotient_gap_within.csv`. Findings:

| dataset | facet | n | r [95% CI] |
|---|---|---:|---|
| composite | noise = 0.0 | 12 | **+0.87** [+0.70, +0.98] |
| composite | noise = 0.2 | 12 | **+0.91** [+0.80, +0.97] |
| composite | noise = 0.4 | 12 | **+0.93** [+0.90, +0.98] |
| wbc | noise = 0.0 | 18 | +0.37 [−0.13, +0.78] |
| wbc | noise = 0.2 | 18 | **+0.50** [+0.10, +0.82] |
| wbc | noise = 0.4 | 18 | **+0.80** [+0.68, +0.93] |
| mnist | per target_dim | 3 | too sparse |

Exp 3 *strengthens* under stratification on composite (cross-cell
+0.77 → within-noise +0.87/+0.91/+0.93). The cross-cell result was
attenuated, not inflated, by per-noise mean shifts. wbc trends
positive in every regime. The composite at noise = 0.4 result
(r = +0.93 [+0.90, +0.98], n = 12) is the strongest single
statistic of the entire series.

## Files touched in this session

**New:**
- `src_experiment/rtg_overlap.py`
- `scripts/rho_func_vs_gen_gap.py`
- `scripts/rl_proxy_vs_quotient_gap.py`
- `figures/rho_func_vs_gen_gap.png`
- `figures/rl_proxy_vs_quotient_gap.png`
- `results/rho_func_vs_gen_gap_correlations.csv`
- `results/rho_func_vs_gen_gap_within.csv`
- `results/rl_proxy_vs_quotient_gap_correlations.csv`
- `results/rl_proxy_vs_quotient_gap_within.csv`
- `logs/exp3_chain_20260427_181714.log`
- `logging/paper_experiments_summary.md`
- `logging/2026-04-27_exp2_exp3_completion.md` (this file)

**Modified:**
- `src_experiment/functional_quotient.py` (rl_proxy column)
- `results/composite_label_noise_new_estimator.csv`
  (refreshed with rl_proxy, training-results columns rejoined)
- `results/wbc_label_noise_new_estimator.csv`
  (refreshed with rl_proxy, training-results columns rejoined)
- `results/mnist_capacity_new_estimator.csv` (trustworthy archs only
  refreshed; non-trustworthy `[15,15,15]`/`[25,25,25]`/`[100,100,100]`
  rows still carry NaN `rl_proxy`)
- `logging/2026-04-27_paper_grounded_experiments.md` (struck the
  "Steps not yet done" section)
- Memory: `project_routing_estimator.md` (status, modules, findings,
  operational gotchas)

## Open follow-ups (not in scope this session)

Listed at the bottom of `paper_experiments_summary.md`:

1. `comp_new_lf_*` GeoLoss ablation — apply the same Exp 1-3
   protocol.
2. MNIST label-noise training sweep (would let Exp 1 noise-suppression
   replicate on mnist).
3. Densify around `[7]³` × PCA ≥ 5 on mnist (capacity-saturation peak).
4. Larger UCI dataset to give wbc a successor with tighter CIs.
5. ~~Within-noise Exp 3 stratification~~ — done in Section 3.1.
