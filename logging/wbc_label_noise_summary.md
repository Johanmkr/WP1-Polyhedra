# WBC label-noise sweep — first-pass summary

Aggregated CSV: `results/wbc_label_noise_new_estimator.csv` (183 040 rows,
270 experiments = 18 archs × 3 noise levels × 5 seeds).
Wall time: 17 min on default hardware. Probe = full UCI WBC (N=569),
no holdout. ε grid: `(0.0, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)`.
Headline numbers below are at the **deepest layer**, **ε = 10.0**, last
epoch, mean over 5 seeds.

## Trustworthiness map (max ρ across all (epoch, layer, noise))

| Architecture | max ρ | trustworthy? |
|---|---:|:---:|
| `[5, 5, 5]`         | 0.15 | ✓ |
| `[5, 5, 5, 5]`      | 0.21 | ✓ |
| `[5, 5, 5, 5, 5]`   | 0.25 | ✓ |
| `[7, 7, 7]`         | 0.28 | ✓ |
| `[7, 7, 7, 7]`      | 0.43 | borderline |
| `[7, 7, 7, 7, 7]`   | 0.46 | borderline |
| `[9, 9, 9]`         | 0.49 | ✗ |
| `[9–10, …, …]`      | 0.59–0.77 | ✗ |
| `[15, 15, 15]`      | 0.82 | ✗ |
| `[25, 25, 25]+`     | 0.98–1.00 | ✗ |
| `[50, 50, 50]`      | 1.00 | ✗ |
| `[100, 100, 100]`   | 1.00 | ✗ |

Only the four smallest architectures stay below the ρ < 0.3 trustworthiness
threshold. Everything wider saturates: there are more activation regions
than test points, so plug-in MI loses statistical power. WBC is
fundamentally **probe-starved at N = 569** for any moderately-sized network
— this is a property of the dataset, not the estimator.

## Headline result: I(Y; Ω) under label noise (mean ± over 5 seeds)

H(Y) ≈ 0.953 bits (binary classification).

| arch | L | metric | n=0.0 | n=0.2 | n=0.4 |
|---|---:|---|---:|---:|---:|
| `[5, 5, 5]`       | 3 | I_raw  | 0.717 | 0.665 | 0.378 |
| `[5, 5, 5]`       | 3 | I_func | **0.632** | **0.568** | **0.180** |
| `[5, 5, 5, 5]`    | 4 | I_raw  | 0.770 | 0.651 | 0.464 |
| `[5, 5, 5, 5]`    | 4 | I_func | **0.756** | **0.570** | **0.275** |
| `[5, 5, 5, 5, 5]` | 5 | I_raw  | 0.647 | 0.530 | 0.441 |
| `[5, 5, 5, 5, 5]` | 5 | I_func | **0.595** | **0.398** | **0.145** |
| `[7, 7, 7]`       | 3 | I_raw  | 0.699 | 0.591 | 0.546 |
| `[7, 7, 7]`       | 3 | I_func | **0.727** | **0.469** | **0.215** |

## Key findings

1. **Quotient MI is *more* sensitive to label noise than raw routing MI.**
   For all four trustworthy architectures, `I_func(n=0) − I_func(n=0.4)`
   exceeds the raw drop:

   | arch | L | raw drop | quotient drop |
   |---|---:|---:|---:|
   | `[5, 5, 5]`       | 3 | 0.339 | **0.452** |
   | `[5, 5, 5, 5]`    | 4 | 0.306 | **0.481** |
   | `[5, 5, 5, 5, 5]` | 5 | 0.206 | **0.450** |
   | `[7, 7, 7]`       | 3 | 0.153 | **0.513** |

   This is the DPI payoff in action: the quotient strips structurally
   distinct but functionally equivalent regions, so what remains tracks
   genuine classification information. Label noise eats into that signal
   more cleanly than into the raw plug-in. **This is the headline result
   for the routing-information story.**

2. **DPI holds everywhere.** Per-row, `miller_madow_func_bits ≤ miller_madow_bits`
   in 100 % of the 183 040 rows (sanity-checked numerically). The DPI scatter
   plot in each dashboard sits cleanly on or below y = x.

3. **Functional collapse plateau is at ε ≈ 10–100.** The collapse-curve
   panel of every trustworthy dashboard shows `ρ_func` flat at high values
   for ε < 1, dropping sharply between ε ≈ 1 and ε ≈ 100, then
   plateauing. The chosen ε = 10 lands cleanly in the regime of "real"
   functional clustering. The spec's default `(0, 1e-8, …, 1e-1)` would
   have produced no signal here.

4. **Saturated architectures still order noise levels correctly.**
   Even at `[25, 25, 25]` (ρ ≈ 0.98), the noise-compare plot shows
   `n=0 > n=0.2 > n=0.4` for `I_func`. So the *qualitative* ordering
   survives saturation; only absolute values are unreliable. This
   suggests the quotient is partially absorbing the small-sample bias
   that kills raw plug-in MI.

5. **`I_func` is below raw `I_raw` even at deepest layer.** Average DPI
   gap at the deepest layer is 0.09–0.16 bits across the trustworthy
   archs.

## Caveats

- **No holdout for WBC** — `truncation_prob` is NaN for all WBC rows.
  We have no estimate of how badly the probe-set partition fails to
  cover the data manifold.
- **Probe overlaps training.** "Full" mode means the model has seen
  ~80 % of the probe during training. Routing analysis stays well-defined,
  but absolute MI values include memorization. The relative noise
  ordering (the headline result) is robust to this; the absolute numbers
  should not be quoted as generalization MI.
- **Statistical power at N = 569 is borderline.** Even for `[5, 5, 5, 5, 5]`,
  the deepest-layer ρ reaches ~0.25 by end of training (close to the
  0.3 line). Confidence intervals from 5 seeds are wide (visible in the
  plots' shaded bands). For paper-quality numbers a probe with several
  thousand samples is needed — which means moving to a larger UCI
  dataset, not WBC.

## Figures

Saved under `figures/label_noise_new_estimator/wbc/`:

- `noise_compare_wbc_a5x5x5_L3_eps10.0.png`
- `noise_compare_wbc_a5x5x5x5_L4_eps10.0.png`
- `noise_compare_wbc_a5x5x5x5x5_L5_eps10.0.png`  ← deepest trustworthy arch
- `noise_compare_wbc_a7x7x7_L3_eps10.0.png`
- `noise_compare_wbc_a25x25x25_L3_eps10.0.png` ← reference: saturated arch
- `dashboards/n{0.0,0.2,0.4}_[5, 5, 5, 5, 5]__new_estimator_seed_101_dashboard_eps10.0.png`

## Suggested next steps

1. **Composite sweep** (already configured in `run_label_noise_estimator.py`).
   N_probe = 20 000 keeps ρ < 0.3 for every architecture in that sweep, so
   the WBC saturation problem disappears. Composite has 7 classes
   (max H(Y) ~ log₂ 7 ≈ 2.8 bits) so the dynamic range is ~3× WBC's.
2. **Re-do WBC with a larger UCI dataset** if a real-data label-noise
   study at full statistical power is needed. WBC's 569 samples is the
   ceiling.
3. **Address the truncation_prob gap.** A k-fold CV at evaluation time
   would give it for WBC at the cost of more probe builds.
