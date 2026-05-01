# Phase A.1 on MNIST + LeNet — extension plan

*Created 2026-04-29. Wiring committed 2026-04-29; ready to sweep via
`./run_mnist_full_lenet_baselines.sh`.*

## Why

The §5.1 baseline-MI write-up (`logging/2026-04-29_baseline_mi_paper_section.md`)
currently rests on two datasets — *composite* (synthetic 2D, 7 classes,
$N = 10\,000$) and *wbc* (UCI binary, $N = 569$). Adding MNIST gives a
third dataset with (a) higher input dimension, (b) a small CNN
backbone, and (c) the noise axis we already need for §6 (Phase C-M3).
This is the cheapest single experiment that converts the baseline
panel from "two MLP-on-tabular datasets" into a paper-defensible
"MLP + CNN, low-d + high-d" comparison.

The trained models already exist: 60 HDF5s under
`outputs/mnist_full_lenet/` (4 archs × 3 noise × 5 seeds, 150 epochs),
landed in commit `d477330` (Phase C-M1, 2026-04-28).

## Scope

This plan covers **only the MI baseline sweep** on the existing
`mnist_full_lenet` HDF5s — i.e. extending Phase A.1 to a new dataset.
It does **not** include the C-M3 functional-quotient sweep (separate
plan, ~30 h compute) and does **not** include re-running gen-gap
predictors (Phase A.2 on MNIST is a separate, cheaper extension).

Done means: `results/mi_baselines.csv` carries an `mnist_full_lenet`
block and `figures/baseline_mi_with_plugin.png` renders three panels
(composite | wbc | mnist).

## Prerequisites already in place

- Trained HDF5s: `outputs/mnist_full_lenet/{n0.0,n0.2,n0.4}_LeNet-{XS,S,M,L}/seed_{101..105}.h5`.
- Activation extractor: `src_experiment/baselines/activations.load_layer_activations`
  is already CNN-aware via the LeNet5 forward path used by
  `src_experiment/cnn_estimator.py` (verified in
  `tests/test_cnn_estimator.py` — activation equivalence test passes).
- All five baselines validated against synthetic ground truth on
  2026-04-28 (12 assertions, max $|\Delta| = 0.021$ bits).
- The plug-in vs Miller–Madow split is already in
  `src_experiment/routing_estimator.routing_information` — no code
  change needed, only the aggregation script needs to surface
  `bits_ours_plugin` alongside `bits_ours_raw`.

## Status (2026-04-29)

Wiring landed in this session:

- `src_experiment/probe_loader.py`: `make_mnist_full_lenet_probe()` reads
  `points`/`labels` from a canonical LeNet HDF5 (10 000 MNIST samples
  already in $[-1, 1]$). Cached.
- `src_experiment/baselines/activations.py`: `load_lenet_layer_activations`
  + `load_activations_dispatch` — instantiates `LeNet5` from HDF5
  metadata, loads `state_dict`, and taps the requested ReLU step's
  pre/post activation via a forward hook. Smoke-tested on LeNet-XS layer 4
  (d_T = 42).
- `run_label_noise_estimator.py`: `mnist_full_lenet` registered in
  `DATASETS`; the directory regex was relaxed to accept either
  `n0.4_[7, 7, 7]` (MLP) or `n0.0_LeNet-XS` (LeNet) layouts.
- `scripts/run_mi_baselines.py`: `_h5_summary` now dispatches on
  `arch_type`; `_resolve_probe` wires in the MNIST probe; `evaluate_one`
  uses the dispatch loader; `_crossref_existing` surfaces both
  `bits_ours_plugin` (plug-in, eq. 8) and `bits_ours_raw` (Miller-Madow,
  eq. 9) plus their `_func` analogues.
- `scripts/plot_routing_vs_baselines.py`: `DATASET_ORDER` lists the
  three datasets; per-dataset `DEEP_LAYER_BY_DATASET` /
  `LAST_EPOCH_BY_DATASET` (5/150 for the MLP runs, 4/100 for LeNet);
  the figure tolerates a missing per-dataset estimator CSV (baselines
  panel without ours-side overlays).
- `run_mnist_full_lenet_baselines.sh`: end-to-end driver (pre-flight
  smoke run -> 60-job sweep -> aggregate -> replot). Executable; logs
  to `logs/mnist_full_lenet_baselines_<timestamp>.log`. Env switches
  documented at the top of the script.

Single-cell wall (LeNet-L, full baselines, 1 MINE seed + 1 InfoNCE
seed): **32 s**. With the runner defaults (5 MINE seeds + 3 InfoNCE
seeds) extrapolating to ~140 s/cell on LeNet-L and ~95 s on LeNet-XS, the
60-cell sweep is **~1.8 h on CPU**. The pre-flight smoke run (cheap
baselines, 1 cell) is ~9 s and is left in the .sh as a default sanity
gate.

## How to run

```bash
# Full sweep (~1.8 h CPU, recommended). Pre-flight + sweep + aggregate
# + replot, all logged to logs/mnist_full_lenet_baselines_<ts>.log.
./run_mnist_full_lenet_baselines.sh

# Cheap variant (~3-5 min): drop the variational baselines.
MNIST_SKIP_MINE=1 MNIST_SKIP_INFONCE=1 ./run_mnist_full_lenet_baselines.sh

# Just list the 60 jobs and exit (no compute).
MNIST_DRY_RUN=1 ./run_mnist_full_lenet_baselines.sh
```

After the sweep, `figures/baseline_mi_with_plugin.png` will render three
panels. The MNIST panel will show all five baselines (with `binning_8`
collapsing to a strongly negative value at d_T = 42 — the §4.5
fine-resolution failure mode the paper claims; KSG/kmeans/InfoNCE/MINE
all behave). The ours-side lines on the MNIST panel will appear once
the C-M3 routing-info estimator sweep lands.

## Steps

### 0. Pre-flight (~5 min)

- `uv run python -c "from src_experiment.baselines.activations import load_layer_activations; import h5py; ..."`
  → load **layer 5** (deepest FC) activations from one
  `outputs/mnist_full_lenet/n0.0_LeNet-XS/seed_101.h5` at the last
  saved epoch. Confirm shape `(N_test, fc_widths[-1])`. If conv layers
  are wanted later, layers 1..4 are 4D and need flattening — defer to a
  follow-up.
- Subsample N: 5000 (matches composite/wbc baseline N). MNIST test set
  is 10 000 so this is straightforward.

### 1. Extend the runner (~30 min)

`scripts/run_mi_baselines.py` currently calls
`run_label_noise_estimator.discover_jobs`, which hard-codes
`DATASETS = {"composite", "wbc"}` (`run_label_noise_estimator.py:48`).
Two minimal options:

- **Option A (preferred).** Add `"mnist_full_lenet": OUTPUTS / "mnist_full_lenet"`
  to that `DATASETS` dict and a thin probe-loader entry that points at
  the MNIST test set (already saved into the HDF5 under `points/`,
  `labels/`). This is mechanically the same wiring as the existing
  `make_composite_probe` / `make_wbc_probe` shims.
- **Option B.** Mirror `run_mnist_capacity_estimator.py` and write a
  parallel `run_mnist_full_lenet_baselines.py`. Heavier, no real
  benefit unless the LeNet activation tap differs from MLP-deepest-FC
  (it does not — layer 5 is the last FC for both).

Pick A. Touchpoints:

- `run_label_noise_estimator.py`: add `mnist_full_lenet` to `DATASETS`;
  add `make_mnist_full_lenet_probe` next to `make_wbc_probe` (just
  reads `points`/`labels` from the HDF5).
- `scripts/run_mi_baselines.py`: nothing — it just iterates over
  whatever `discover_jobs` returns, and the `run_id` keying already
  carries `dataset`.
- **Surface the plug-in.** While in `scripts/run_mi_baselines.py:113`,
  also write `bits_ours_plugin = float(raw["plug_in_bits"])` and the
  `_func` analogue. This is a one-line change but it's what unblocks
  the plug-in column in the figure for *all three* datasets, so do it
  now and reaggregate composite + wbc as a freebie.

### 2. Sweep (~3–6 h on CPU)

```bash
# From repo root.
uv run python scripts/run_mi_baselines.py \
    --datasets mnist_full_lenet \
    --noise 0.0 0.2 0.4 \
    --layer 5
```

- Per-cell wall on the small composite networks was ~43 s when MINE
  was on (mostly MINE). LeNet-XS layer-5 activations are
  $\dim = 84$ (LeNet-5 last FC), comparable to composite's 25-wide
  layer-5; expect ~60 s/cell. 60 cells × 60 s × overhead ≈ ~1.5–2 h
  for the four LeNet sizes. Worst case (LeNet-L, $\dim$ wider) ~6 h.
- Output: per-HDF5 `mi_baselines_seed_<seed>.csv` next to each
  `seed_*.h5`, plus `--aggregate` to roll into
  `results/mi_baselines.csv`.

### 3. Aggregate + reaggregate (~2 min)

```bash
uv run python scripts/run_mi_baselines.py --aggregate
# -> results/mi_baselines.csv now has composite + wbc + mnist_full_lenet
```

- If step 1's plug-in column was added, also rerun the composite + wbc
  aggregation so that block carries `bits_ours_plugin` too.

### 4. Replot (~5 min)

```bash
uv run python scripts/plot_routing_vs_baselines.py
```

- Update the script (one-line; see comment in the script) to add
  `'mnist_full_lenet'` to `DATASET_ORDER`, then it autopanels.
- Output: `figures/baseline_mi_with_plugin.png` with three panels.

### 5. Update the §5.1 write-up (~10 min)

Edit `logging/2026-04-29_baseline_mi_paper_section.md`:

- Add the mnist row(s) to the Results table.
- Update the "What this experiment does not yet cover" section: drop
  the MNIST bullet, keep the Phase B and critic-capacity bullets.
- Skim the three observations — particularly observation 1 (bias
  correction matters): MNIST is the natural high-$|\Omega|$
  arena to test that claim, since LeNet has many more parameters than
  the MLPs and the routing-information support will be larger relative
  to $N$.

## Risks and how to handle

- **LeNet conv layers in the activation tap.** Layer-5 (last FC) is
  fine and matches the composite/wbc deepest-layer convention. Conv
  layers (layers 1–3 in the LeNet HDF5) need flattening + a decision
  about whether per-channel or per-spatial-cell binning is the right
  baseline. Defer; the figure works on the FC tap alone.
- **MINE on 84-d activations.** The MLP was 25-d and MINE was already
  the dominant cost. If wall blows past 6 h on LeNet-L, fall back to
  `--skip-mine` for the first pass and revisit. Plug-in / Miller–Madow
  / KSG / InfoNCE / binning all still run.
- **Probe-loader edge case.** `outputs/mnist_full_lenet/*.h5` may have
  been saved with a different `points/labels` schema than
  `outputs/composite_label_noise/`. Confirm in pre-flight; if so, the
  `make_mnist_full_lenet_probe` shim absorbs the difference rather
  than touching baseline code.
- **`run_id` collisions in `mi_baselines.csv`.** The `network_id`
  string already includes `dataset` (`run_label_noise_estimator.py:97-ish`),
  so `mnist_full_lenet` rows will not collide with the composite block.

## When to revisit

After §5.1 + figure update lands, the natural follow-up is **Phase
A.2 on MNIST** (gen-gap predictors against the same 60 HDF5s). That is
~1 h compute and lets the §5.2 Kendall-τ table also extend to three
datasets. Out of scope for this plan.

## Done definition

- [ ] `results/mi_baselines.csv` contains `mnist_full_lenet` rows for
      $4 \text{ archs} \times 3 \text{ noise} \times 5 \text{ seeds} = 60$ cells.
- [ ] `bits_ours_plugin` (and `bits_ours_func_plugin`) columns present
      across the *whole* CSV (composite + wbc reaggregated).
- [ ] `figures/baseline_mi_with_plugin.png` renders three panels.
- [ ] `logging/2026-04-29_baseline_mi_paper_section.md` Results table
      has a third dataset block, and the "MNIST" item is removed from
      the not-yet-covered list.
- [ ] One-paragraph addendum to §5.1 ("the bias-correction gap is
      largest where Theorem 4.5's regime applies", supported by the
      MNIST/LeNet-L row at $\eta = 0.4$).

## To resume later today

```bash
# from repo root
cat planning/phase_a_mnist_baselines.md
# step 0 (pre-flight) — verify activation tap shape, then steps 1..5.
```
