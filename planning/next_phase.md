# Next phase — paper-grounded follow-ups

The composite label-noise sweep, MNIST capacity Phase A+B, and
Experiment 1 (training dynamics) are **done**. Logs:

- `logging/2026-04-27_composite_sweep_run.md`
- `logging/composite_label_noise_summary.md`
- `logging/mnist_capacity_summary.md`
- `logging/2026-04-27_paper_grounded_experiments.md`  ← Step 1 + Step 2 of the
  paper-grounded experiment series

This file is now the entry point for picking up the paper-grounded
experiment series (Sec 4–6 of `Johan_binning_paper_NeurIPS_2026-1.pdf`).
Read the log first to see what shape the data is in.

## Status (2026-04-27)

- **Aggregated CSVs carry training-results columns** (`train_acc`,
  `test_acc`, `train_loss`, `test_loss`, `gen_gap_acc`, `gen_gap_loss`)
  for composite, WBC and MNIST. Backups in `*.bak`.
- **Training-dynamics figures** for composite/wbc/mnist saved to
  `figures/training_dynamics/`. The headline finding —
  **compression-then-expansion in ρ_func** mirroring the IB phase
  transition, suppressed by label noise — is documented in the log.
  This finding is paper-relevant (Sec 6).

## Phase goal

Finish Experiments 2 and 3 of the paper-grounded series, then write a
single short summary that combines the dynamics finding with the
ρ_func-as-diagnostic and routing-loss-proxy results.

## Step-by-step

### 3. Experiment 2 — ρ_func ↔ generalization gap

Because `train_acc` in noise sweeps is computed against *noisy* labels,
the raw `gen_gap_acc` column is misleading (negative under noise). Use
one of:

- **Noise-adjusted train accuracy.** Best clean train_acc the model
  could achieve under noise level n is `1 − n` (memorizing perfectly on
  the corrupted set means matching every label, which equals the true
  label only `1 − n` of the time). Define `gen_gap_norm =
  train_acc − (1 − noise_level) · test_acc`.
- **Absolute gap.** `|train_acc − test_acc|` is a simpler proxy that
  doesn't need the noise model.

Plot scatter of `gen_gap_norm` vs `ρ_func(deepest, ε=10, last epoch)`
mean over seeds, colored by arch and faceted by dataset. Report
correlation coefficient with bootstrap CI.

Hypothesis: ρ_func near 1 ⇔ "near-full combinatorial use" ⇔ network
memorizes ⇔ wide gap. Small ρ_func ⇔ structured computation ⇔ tighter
gap. Composite at noise = 0.4 should land high-ρ_func / wide-gap;
composite at noise = 0 / `[5,5,5]` should land low-ρ_func / tight-gap.

Output: `figures/rho_func_vs_gen_gap.png`, summary section in
`logging/paper_experiments_summary.md`.

Cost: ~1 hour, no new compute.

### 4. Experiment 3 — RTG-aware routing-loss proxy

This *does* need new compute and a small new module.

**Plan:**

1. New module `src_experiment/rtg_overlap.py` exposing
   `routing_loss_proxy(activation_patterns, labels) -> float`. Returns
   the fraction of Hamming-1-adjacent region pairs (data-supported,
   layer-l) whose dominant-class endpoints differ. Sec 3.2 of the paper
   defines the data-supported RTG; this is the natural finite-data
   surrogate for Prop 4.2.
2. Wire it into `FunctionalQuotientEstimator.evaluate_all` as a new
   column `RL_proxy`. Compute per (epoch, layer); does not depend on ε.
3. Re-run the estimator on:
   - composite (180 HDF5s; ~47 min).
   - wbc (~270 HDF5s; ~17 min).
   - mnist trustworthy subset (`[3,5,7]³` × all PCA dims × all seeds;
     ~80 jobs but mostly fast — should be ~10 min).
4. Re-aggregate.
5. Plot scatter of `Ĩ_raw − Ĩ_func` vs `RL_proxy` per row, colored by
   noise level. Hypothesis: positive correlation (Prop 4.2 says
   RL_proxy > 0 is necessary for routing-loss > 0; the I_raw − I_func
   gap is an upper bound proxy on the routing loss).

Cost: ~3 h coding + ~1.5 h compute.

### 5. Combined summary

`logging/paper_experiments_summary.md`:

- Section 1: training dynamics (compression-then-expansion in ρ_func,
  4–6 panel figure).
- Section 2: ρ_func ↔ gen-gap scatter + correlation table per dataset.
- Section 3: RL_proxy ↔ I_raw − I_func scatter, validating Prop 4.2.
- Section 4: cross-dataset replication table (which findings transfer
  across composite, wbc, mnist?).

This is the artefact that maps onto Sec 6 of the paper.

## Open items deferred

These remain unchanged from the previous phase plan:

- Unapplied Miller-Madow correction in
  `src_experiment/estimate_quantities.py:271` (legacy `ExperimentEvaluator`).
  Trivial one-line fix; left untouched per the original "no edits to
  existing modules" rule.
- LSH for Recipe 2 if region count exceeds ~10 k. Not relevant for any
  current sweep.
- `truncation_prob` for WBC. Would require k-fold CV at evaluation
  time. Not in scope.

## Useful reference paths

- Drivers: `run_label_noise_estimator.py`, `run_mnist_capacity_estimator.py`.
- Probe loader: `src_experiment/probe_loader.py`.
- Plotter: `visualization/plot_new_estimator.py`.
- Estimator core:
  `src_experiment/{routing_estimator,functional_quotient,rtg_analyzer}.py`.
- New helper scripts:
  - `scripts/composite_sanity_check.py`
  - `scripts/cross_dataset_comparison.py`
  - `scripts/mnist_capacity_landscape.py`
  - `scripts/join_training_results.py`
  - `scripts/training_dynamics.py`

## When you're done

The natural follow-ons (out of scope here):

- **`comp_new_lf_*` ablation.** The geometric-loss training ablation in
  `outputs/comp_new_lf_*` is the place where the new estimator most
  likely shows direct scientific value: with vs without the GeoLoss
  term, does the quotient MI separate the two regimes?
- **MNIST label-noise training sweep.** Composite memorization
  signature was visible because of the noise axis. To replicate it on
  MNIST would need a fresh training sweep with noise ∈ {0, 0.2, 0.4} on
  3–4 archs at PCA ∈ {5, 20}.
- **Larger UCI dataset.** WBC's N = 569 is the ceiling. A
  middle-ground UCI dataset (a few thousand samples, 5–10 classes)
  would give a third anchor between WBC's small-N regime and
  composite's big-N regime.
- **Densify around `[7]³` on MNIST.** Phase A+B revealed `[7]³` × PCA
  ≥ 5 as the class-info sweet spot. Widths {6, 7, 8, 9, 10, 12} at
  PCA ∈ {5, 10, 20} would map the capacity peak more precisely.
  Needs new training runs.

## Useful paper sections to refer back to

- **Theorem 4.1** — the four-term decomposition I(Y;T) = I(Y;Π) +
  I(Y;T|Π) − I(Y;Π|T). The estimator only sees I(Y;Π) directly; Exp 3
  attacks I(Y;Π|T).
- **Theorem 4.5 + chain (10)** — Ĩ ≤ I(Y;Π) ≤ I(Y;X) ≤ H(Y).
  Already verified empirically (DPI clean across all sweeps).
- **Proposition 4.2** — routing loss positive iff overlap +
  label-disagreement. Geometric reading in App E.4. Exp 3 builds an
  empirical surrogate.
- **Proposition 4.6** — quotient is a tight LB; equality whenever
  downstream is deterministic in (A_ω, c_ω). Exp 1's "memorization via
  redundant routing" finding is the empirical face of this proposition.
- **Section 4.5** — the fine-resolution problem; ρ_func as a
  diagnostic. Exp 2 connects this diagnostic to generalization.
