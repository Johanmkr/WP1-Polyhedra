# Label-noise experiment — new-estimator evaluation plan

Run `FunctionalQuotientEstimator.evaluate_all` (Recipes 1–4) over the
already-trained label-noise sweeps in
`outputs/composite_label_noise/` and `outputs/wbc_label_noise/`.
No new training is required.

## What's available

| Dataset | Test N | Classes | Input dim | Architectures | Noise | Seeds | HDF5s |
|---|---:|---:|---:|---|---|---|---:|
| `composite_label_noise` | 2000 | 7 | 2 | 12 (`[5..25]^{3,4,5}`) | 0.0, 0.2, 0.4 | 101–105 | 180 |
| `wbc_label_noise` | 114 | 2 | 30 | 18 (incl. `[10..]`, `[15..]`, `[50..]`, `[100..]`) | 0.0, 0.2, 0.4 | 101–105 | 270 |

22 epochs saved per HDF5 (`save_epochs: 0..10` plus every 10 to 150).

## Estimator config

- ε grid (wider than spec default): `(0.0, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)`.
  Spec defaults `(0, 1e-8, ..., 1e-1)` don't collapse anything in trained nets — natural
  separation scale is 10–100.
- Probe / holdout policy (composite vs. wbc differ — see "Probe sets" below).
- Output: one CSV per HDF5 next to the file as `new_estimator_seed_<seed>.csv` —
  idempotent (skip if exists), so the run can be interrupted/resumed.
- Each row = `(network_id, epoch, layer, ε, …)` with the full schema from
  `logging/new_estimator_implementation.md` §8, plus `dataset`, `noise_level`,
  `arch_str`, `probe_N` columns added by the driver.

## Probe sets

`src_experiment/probe_loader.py` builds (and caches) probe + holdout bundles
that live in the same preprocessed feature space as the trained model.

### Composite — regenerate fresh

`_make_composite_data` is fully synthetic, so we rebuild the training-time
scaler from `(global_seed=42, N_SAMPLES=10000)` and apply it to fresh samples
generated with distinct seeds for probe and holdout. Labels are kept clean
(label noise was injected on training labels only — for the routing-MI study
we want $I(Y_{\text{true}};\Omega)$). Activated via:

```bash
--composite-probe-size 20000 --composite-holdout-size 10000
```

Verified end-to-end on `[25, 25, 25, 25, 25]` seed 101: ρ stays < 0.11
across all 5 layers, `truncation_prob` ranges 0.0008–0.027, DPI holds
everywhere. Wall time: **~59 s** for the largest architecture
(180 jobs × ~30–60 s ⇒ 1.5–3 h for the full composite sweep).

### WBC — three modes via `--wbc-mode`

Only **569 total samples** in WBC, so all options have caveats:

| Mode | Probe N | Holdout | Notes |
|---|---:|---:|---|
| `test` (default) | 114 | – | stored test set; ρ blows past 0.3 on most arches |
| `full` | 569 | – | full UCI WBC; train portion seen during training but routing analysis is well-defined; **recommended** |
| `split` | 80 | 34 | stratified 70/30 split of stored test; only useful for `[5,5,5]` / `[7,7,7]` |

`full` mode tested on `[5,5,5]`/n0.0/seed_101: ρ ∈ [0.02, 0.09], 0.57 s/job.
Full WBC sweep at `full` mode is minutes, not hours.

### Caveats

- **No truncation_prob for WBC `full` or `test`** — there's no held-out set.
- **WBC at N=569 with `[100,100,100]`** will still hit ρ > 0.3. The `rho`
  column shows it; filter post hoc.

## Run plan

Phase 1 — **timing one experiment** (already verified):

```bash
uv run python run_label_noise_estimator.py --datasets composite --limit 1 \
    --composite-probe-size 20000 --composite-holdout-size 10000
# ~59 s on [25, 25, 25, 25, 25]; smaller archs proportionally faster
```

Phase 2 — **full composite sweep** (recommended next):

```bash
uv run python run_label_noise_estimator.py --datasets composite \
    --composite-probe-size 20000 --composite-holdout-size 10000
# 180 HDF5s, ETA ~1.5–3 h
```

Narrow architecture filter for a faster pass first:
```bash
uv run python run_label_noise_estimator.py --datasets composite \
    --composite-probe-size 20000 --composite-holdout-size 10000 \
    --archs '[5, 5, 5]' '[9, 9, 9]' '[25, 25, 25]'
```

Phase 3 — **WBC sweep** (`full` mode recommended):

```bash
uv run python run_label_noise_estimator.py --datasets wbc --wbc-mode full
# 270 HDF5s, ETA ~minutes
```

Phase 4 — **aggregate**:

```bash
uv run python run_label_noise_estimator.py --aggregate \
    --output results/label_noise_new_estimator.csv
```

Concatenates every `new_estimator_seed_*.csv` under
`outputs/{composite,wbc}_label_noise/**/` plus tags `dataset`,
`noise_level`, `arch_str` from the directory layout.

## Analysis suggestions (after CSV is built)

1. **Sanity:** `plug_in_func_bits ≤ plug_in_bits` (DPI must hold per row).
2. **ρ check:** filter `rho < 0.3` before drawing scientific conclusions.
3. **Noise-effect plot:** for each `(dataset, arch)`, plot
   `miller_madow_func_bits` (best layer, large ε) vs. epoch, faceted by
   noise level. Expectation under the routing-MI story: with label noise,
   the quotient MI should track the *clean* signal more faithfully than
   raw plug-in MI.
4. **Quotient collapse curve:** plot `rho_func` vs. ε per layer. The flat
   plateau ~ ε ∈ [10, 100] should be the "true" functional cardinality.
5. **RTG geometry vs. noise:** `rtg_largest_component_frac` and
   `rtg_isolated_frac` over training; does label noise leave more isolated
   regions?

## Files in this rollout

- `run_label_noise_estimator.py` — driver (this is the script).
- `new_estimator_seed_<seed>.csv` — per-HDF5 outputs (alongside HDF5).
- `results/label_noise_new_estimator.csv` — aggregated CSV from `--aggregate`.
