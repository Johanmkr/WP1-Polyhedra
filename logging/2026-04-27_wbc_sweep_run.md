# 2026-04-27 — WBC label-noise sweep run

Session ran the full label-noise estimator sweep on
`outputs/wbc_label_noise/` (270 HDF5s = 18 archs × 3 noise × 5 seeds).
Composite sweep deliberately deferred — WBC is the smaller, cheaper run
to validate the pipeline before committing to the longer composite run.

## What was added this session

### New code
- **`run_label_noise_estimator.py`** (repo root) — driver. Walks
  `outputs/{composite,wbc}_label_noise/`, calls
  `FunctionalQuotientEstimator.evaluate_all` per HDF5, writes
  `new_estimator_seed_<seed>.csv` next to each. Idempotent, resumable, with
  per-job timing + ETA. CLI flags filter by dataset / noise / arch / seed
  and control probe policy (composite probe size, WBC mode).
  `--aggregate` concatenates everything into a single CSV.
- **`src_experiment/probe_loader.py`** — `lru_cache`'d probe + holdout
  builders.
  - `make_composite_probe` rebuilds the training-time `MinMaxScaler` by
    replaying `_make_composite_data(N_SAMPLES=10000, seed=42)` → 80/20
    stratified split → fit, then generates fresh probe + holdout with
    distinct seeds and pushes them through the same scaler. Labels stay
    clean (label noise was injected on train labels only).
  - `make_wbc_probe(mode={"test","full","split"})` covers stored test,
    full UCI 569-sample set, or 70/30 split of stored test.
- **`visualization/plot_new_estimator.py`** — single-experiment 2 × 3
  dashboard (MI trajectories, ρ check, ρ_func collapse curve, RTG
  fractions, truncation_prob, DPI scatter) and aggregate-mode noise-compare
  plot (mean ± std band over seeds).

### New artefacts
- 270 per-HDF5 CSVs in `outputs/wbc_label_noise/*/new_estimator_seed_*.csv`.
- `results/wbc_label_noise_new_estimator.csv` — 183 040 rows aggregated.
- 5 noise-compare plots + 3 single-experiment dashboards in
  `figures/label_noise_new_estimator/wbc/`.
- `logging/wbc_label_noise_summary.md` — full results writeup.

### Logs / docs reorganized
- `logging/` and `planning/` directories created at repo root.
- Moved into `logging/`: `new_estimator_implementation.md`,
  `compatability_analysis.md`, `wbc_label_noise_summary.md`, this file.
- Moved into `planning/`: `new_estimator_next_steps.md`,
  `label_noise_experiment_plan.md`, `next_phase.md` (the master entry
  point for the composite continuation).
- Cross-references in moved files updated to the new paths.

## Headline results (full detail in `logging/wbc_label_noise_summary.md`)

- **Quotient MI is *more* sensitive to label noise than raw routing MI.**
  Drop from `n=0` to `n=0.4` is **0.45–0.51 bits** for `I_func` vs
  **0.15–0.34 bits** for raw `I_raw` across all four trustworthy archs
  (`[5,5,5]`, `[5,5,5,5]`, `[5,5,5,5,5]`, `[7,7,7]`). This is the routing-
  information story working as predicted.
- **DPI holds in 100 %** of 183 040 rows.
- **Functional collapse plateau ε ∈ [10, 100]** in every dashboard —
  spec's default `(0, 1e-8, …, 1e-1)` would have produced no signal.
- **WBC is probe-starved at N=569** — only the four smallest archs keep
  ρ < 0.3. Wider archs saturate (ρ → 1). Saturated archs still order
  noise levels correctly but absolute numbers are unreliable.

## Verified properties

- Composite probe builder verified end-to-end on `[25]^5` seed_101 with
  N_probe = 20000, N_holdout = 10000: ρ ≤ 0.106 across all 5 layers,
  truncation_prob ∈ [0.0008, 0.027], DPI holds, 59 s wall-time. (Test
  CSV at `outputs/composite_label_noise/n0.0_[25, 25, 25, 25, 25]/new_estimator_seed_101.csv`
  remains in place.)
- WBC sweep ran in 17 min total, 0 failures.

## Implementation notes worth remembering

- WBC sweep used `--wbc-mode full`, not `--wbc-mode test`. The `full` mode
  includes data the model trained on (~80 % of probe). Routing analysis
  stays well-defined; absolute MI values include memorization. The
  *relative* noise ordering (the headline result) is robust to this; the
  absolute numbers are not generalization MI.
- `truncation_prob` is NaN for the WBC sweep — there's no held-out set in
  the `full` mode by design.
- Plot-script filenames now include the architecture tag for noise-compare
  (`noise_compare_wbc_a5x5x5x5x5_L5_eps10.0.png`) and the parent dir for
  single-experiment dashboards
  (`n0.0_[5, 5, 5, 5, 5]__new_estimator_seed_101_dashboard_eps10.0.png`).
  Earlier filenames collided across runs.

## Pointers for any continuation

- `planning/next_phase.md` — actionable plan for the composite sweep (the
  natural continuation).
- `logging/new_estimator_implementation.md` — full implementation log
  (estimator math, modules, verified properties, open items §12).
- `logging/wbc_label_noise_summary.md` — full WBC results.
