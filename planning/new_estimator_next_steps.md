# Next steps — pick up here

State at end of session 2026-04-26: Recipes 1–4 implemented and verified
(see `logging/new_estimator_implementation.md`). Bias recursion confirmed against
your Eq. 3 algebraically and numerically (~3e-6 fp error). Ready to run the
unified driver on real experiments.

## Decision needed from you

Pick the (dataset, architecture, seed) combinations to evaluate. The plan
below assumes 3–6 datasets × 1–2 architectures × 3 seeds. Adjust as you
prefer.

## Dataset candidates (from your `dataset.py`)

Ranked by likely scientific payoff for the new estimator:

| Dataset | Why it's interesting | Notes |
|---|---|---|
| `blobs` (2d, 5d, 10d) | Clean cluster structure, multiclass, dimension sweep available | ρ controllable via `centers` / `target_dim`; lots of existing runs |
| `moons` | Binary, nonlinear boundary, classic | Small intrinsic complexity → small ρ |
| `wbc` | Real UCI, binary, ~30 features → PCA | Tests behaviour on real-but-simple data |
| `composite` | 5-class moons+circles+blobs mixture, supports label noise | Good for label-noise vs. routing-MI study |
| `mnist_minimal` | 7×7 = 49 features (after avg-pool), 10 classes | Scale stress test; likely ρ > 0.3 unless you grow N |
| `comp_new_lf` vs. `comp_new_lf_NOTGEOLOSS` | Your geometric-loss ablation | Where the new estimator likely shows scientific value |

Avoid first pass: full MNIST (untransformed), `mnist_minimal_random` (no
class signal by design — useful only as a negative control later).

## Architecture suggestions

Existing experiments mostly use `[25, 25, 25]`. For the new estimator:

- **Small reference**: `[10, 10]` — keep ρ low, easy to interpret.
- **Standard**: `[25, 25, 25]` — matches your existing experiments; comparable.
- **Wider/deeper** (optional, only if ρ stays sane): `[50, 50, 50]` or `[25, 25, 25, 25]`.

For each architecture, pick 3 model seeds (e.g. 101, 202, 303) so you can
report mean ± std. Keep `global_seed` fixed across the seed sweep so the
data split is identical.

## Probe-set sizing

The estimator becomes untrustworthy at ρ = `num_regions / N` > 0.3. Existing
runs at N=2000 already hit ρ > 0.3 on blobs_5d+ (per the prior analysis).
Two options:

1. **Increase N**: bump `N_SAMPLES` in `dataset.py` (currently 10000) and
   widen the test split, or add a "probe-only" larger sample regenerated at
   evaluation time using the saved dataset config.
2. **Reduce architecture width** so `num_regions` stays small enough.

Suggestion: keep `[25, 25, 25]` and target N ≥ 5000 in the test split for
2d/5d datasets, N ≥ 10000 for higher dimensions. Confirm by checking the
`rho` column after running.

## Checkpoint schedule

For each training run, save at least:
`save_epochs: [0, 1, 2, 5, 10, 20, 50, 100]` (or your equivalent).

Earlier checkpoints capture init / early-training behaviour where ρ_func
behaviour is most informative.

## Held-out set for truncation probability

The current HDF5 stores only one test split. For each chosen experiment,
either:

- Re-generate the dataset with a different `global_seed` to produce a
  disjoint validation set, then pass `X_holdout, y_holdout` to
  `evaluate_all`.
- Or split the test set into probe (70 %) / holdout (30 %) once at
  evaluation time. Less rigorous but quick.

## ε grid

Spec defaults `(0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1)` collapse nothing on
trained nets — the natural separation scale is 10–100. Use a wider sweep:

```python
epsilons = (0.0, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
```

This both catches the fp-noise regime (low end) and the genuine functional
collapse (high end), so the plateau-finding workflow becomes possible.

## Concrete plan (when you resume)

1. **Confirm dataset list and architectures.** Edit a list at the top of a
   driver script — see the sketch below.
2. **Train (or reuse existing checkpoints).** For experiments that already
   exist in `outputs/`, just point at the HDF5. For new ones, run
   `./run_pipeline.sh <config.yaml>` (Julia step is unnecessary for the new
   estimator but doesn't hurt).
3. **Run `FunctionalQuotientEstimator.evaluate_all` per HDF5.** Stack the
   resulting DataFrames with `pd.concat`; one CSV per study.
4. **Sanity-check the output**: ρ < 0.3 on most rows, plug_in_func ≤ plug_in,
   miller_madow_func behaviour across ε (look for the predicted increase
   with mild ε).
5. **Plot**: per-epoch trajectories of `miller_madow_bits` and
   `miller_madow_func_bits` per layer, faceted by dataset and architecture.
6. **Decide whether to address the open items** in
   `logging/new_estimator_implementation.md` §12 (MM correction in legacy
   `ExperimentEvaluator`, LSH for Recipe 2, etc.) once the data tells you
   what's worth fixing.

## Driver-script sketch

Save as something like `run_new_estimator.py` and edit the `experiments`
list to taste:

```python
from pathlib import Path
import pandas as pd
from src_experiment.functional_quotient import FunctionalQuotientEstimator

experiments = [
    "outputs/blobs_2d/blobs_2d.h5",
    "outputs/blobs_5d/blobs_5d.h5",
    "outputs/wbc_big/wbc_big.h5",
    # ... add/remove as you decide
]

epsilons = (0.0, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

frames = []
for h5 in experiments:
    if not Path(h5).exists():
        print(f"skipping (missing): {h5}")
        continue
    print(f"evaluating {h5}")
    est = FunctionalQuotientEstimator(h5)
    df = est.evaluate_all(epsilons=epsilons)
    df["source"] = h5
    frames.append(df)

result = pd.concat(frames, ignore_index=True)
result.to_csv("results/new_estimator.csv", index=False)
print(f"saved {len(result)} rows")
```

A multi-seed loop is identical: iterate over seed-suffixed HDF5 files
(`outputs/<exp>/seed_<n>.h5`), tag each frame with the seed, concat.

## Hard-stop / context-saving notes

The implementation status, math, and empirical findings are persisted in:

- `logging/new_estimator_implementation.md` — full implementation log.
- `logging/compatability_analysis.md` — original six-question codebase analysis.
- `claude_new_estimator_instructions.md` — original spec (in repo root).
- `~/.claude/projects/-home-johan-Documents-phd-WP1-geometric-binning-estimator/memory/` — Claude's persistent project memory.

If a new session needs to pick up: read this file plus
`logging/new_estimator_implementation.md`. Everything else is reconstructable from
those two.
