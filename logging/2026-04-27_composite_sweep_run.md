# 2026-04-27 — composite label-noise sweep run

## Outcome

- 180/180 jobs completed (179 ran + 1 skipped pre-existing).
- Wall time **47.25 min**, mean **15.84 s/job**. Faster than the
  1.5–3 h estimate in `planning/next_phase.md`.
- Aggregated CSV: `results/composite_label_noise_new_estimator.csv`
  (126 720 rows; 12 archs × 3 noise × 5 seeds × 22 epochs ×
  5 layers × 8 ε ≈ 126 720).
- Per-HDF5 logs: `logs/composite_sweep_20260427_083340.log`.

## Command run

```bash
uv run python run_label_noise_estimator.py --datasets composite \
    --composite-probe-size 20000 --composite-holdout-size 10000
uv run python run_label_noise_estimator.py --aggregate \
    --output results/composite_label_noise_new_estimator.csv
# then composite-only filter (the aggregator pulls both wbc + composite
# per-HDF5 CSVs by default; we filter to keep the file name honest):
python -c "
import pandas as pd
df = pd.read_csv('results/composite_label_noise_new_estimator.csv')
df[df['dataset']=='composite'].to_csv(
    'results/composite_label_noise_new_estimator.csv', index=False)"
```

## Sanity diagnostics (`scripts/composite_sanity_check.py`)

| check | result |
|---|---|
| plug-in DPI (`plug_in_func_bits ≤ plug_in_bits`) | **clean (0/126720 violations)** |
| MM DPI (`miller_madow_func_bits ≤ miller_madow_bits`) | 11 155 / 126 720 (≈8.8 %) violations — expected; MM correction depends on R, so the inequality does not survive bias correction. Not a real DPI failure. |
| max ρ across all rows / archs | 0.170 (`[25,25,25,25,25]`); all 12 archs trustworthy at ρ < 0.3 |
| `truncation_prob` non-null | True for all rows; mean 0.0033, max 0.0612 |
| collapse plateau | ε ∈ [10, 100], same as WBC |

## Headline scientific findings

1. **Quotient noise-sensitivity replicates and amplifies on composite**
   for the 9 narrow archs ([5–9]^{3..5}). Quotient drop > raw drop in
   all 9. Several raw drops are *negative* (network memorizes noisy
   labels into more routing regions, which inflates raw MI under
   noise) — the quotient strips that artefact and exposes a real drop
   in classification information.
2. **Wide [25]^k archs flip the sign on composite.** Both raw and
   quotient MI rise (or stay flat) under noise. ρ stays low (so this
   is not estimator saturation), making it a clean **memorization
   signature**: capacity is large enough to encode noisy labels into
   routing structure. WBC could not show this because [25]^k was
   already saturated at N = 569.
3. **Cross-dataset replication: 5/6 of the shared (arch, dataset)
   cells have quotient drop > raw drop.** Only composite-`[25, 25, 25]`
   breaks the pattern, with both drops negligible — consistent with
   the memorization story.

Full writeup: `logging/composite_label_noise_summary.md`.

## Artefacts

- Aggregated CSV: `results/composite_label_noise_new_estimator.csv`
- Noise-compare plots: `figures/label_noise_new_estimator/composite/`
- Per-experiment dashboards (n=0/0.2/0.4 at `[25,25,25]`, seed 101):
  `figures/label_noise_new_estimator/composite/dashboards/`
- Cross-dataset bar chart:
  `figures/label_noise_new_estimator/cross_dataset/noise_drop_comparison.png`
- Helper scripts (kept for reuse):
  - `scripts/composite_sanity_check.py`
  - `scripts/cross_dataset_comparison.py`

## Loose ends / suggested follow-ups

- Capacity-vs-noise sweep (widths 5..25 at depth 3) to map the
  memorization threshold cleanly.
- Apply the same estimator to `comp_new_lf_*` outputs (geometric-loss
  ablation) — the next high-value experiment for the routing-information
  paper.
- **MNIST capacity follow-up** picked up next (separate plan in
  `planning/`).
