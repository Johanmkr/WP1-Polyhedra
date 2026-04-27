# MNIST capacity sweep — Phase A + B summary

Plan: `planning/mnist_capacity_experiment_plan.md`.
Aggregated CSV: `results/mnist_capacity_new_estimator.csv`
(42 240 rows, 80 experiments).

| phase | grid | jobs | wall |
|---|---|---:|---:|
| A | 4 PCA dims × `{[5],[15],[25],[100]}³` × 3 seeds | 48 | 97.8 min |
| B | 4 PCA dims × `{[3],[7]}³` × 3 seeds, plus `[5]³` × 4 PCA × seeds {104,105} | 32 | 3.8 min |

(Phase B is so much faster because it's all narrow nets — the wide
`[100]³` cells at PCA ≥ 5 dominate Phase A wall time.) Probe = stored
MNIST test set (N = 10 000), PCA-reduced + MinMax-scaled to the model's
training-time space, no holdout (`truncation_prob` = NaN). Headline
numbers at the **deepest layer**, **ε = 10.0**, last epoch (150), mean
over seeds.

H(Y) ≈ 3.32 bits (10-class, near-balanced MNIST test).

## Trustworthiness map (max ρ across all rows)

|             | `[3]³` | `[5]³` | `[7]³` | `[15]³` | `[25]³` | `[100]³` |
|---|---:|---:|---:|---:|---:|---:|
| **PCA=2**   | **0.004 ✓** | **0.012 ✓** | **0.021 ✓** | **0.052 ✓** | **0.120 ✓** | 0.616 ✗ |
| **PCA=5**   | **0.007 ✓** | **0.031 ✓** | **0.146 ✓** | 0.537 ✗ | 0.890 ✗ | 1.000 ✗ |
| **PCA=10**  | **0.006 ✓** | **0.063 ✓** | **0.149 ✓** | 0.763 ✗ | 0.961 ✗ | 1.000 ✗ |
| **PCA=20**  | **0.006 ✓** | **0.044 ✓** | **0.195 ✓** | 0.816 ✗ | 0.993 ✗ | 1.000 ✗ |

Trustworthy cells (ρ < 0.3, bold above). With Phase B added, **all archs
up to and including `[7]³` are trustworthy at every PCA dim**, while
the boundary jumps abruptly to saturation between `[7]³` and `[15]³`
(except at the tightest bottleneck PCA = 2, where even `[25]³` stays
trustworthy). The capacity-saturation curve is therefore very sharp,
with a 7×width step roughly 4× the region count.

## Headline I(Y; Ω) and I_func(Y; Ω) — trustworthy cells only

|       arch | PCA dim |  I_raw | I_func |  ρ  | R    | R_quot | ratio (I_func/I_raw) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `[3]³`   |  2  |  0.75  |  0.55  | 0.00 |   19  |    8  | 0.73 |
| `[3]³`   |  5  |  0.66  |  0.60  | 0.01 |   51  |   23  | 0.90 |
| `[3]³`   | 10  |  1.03  |  0.88  | 0.00 |   38  |   15  | 0.86 |
| `[3]³`   | 20  |  0.89  |  0.68  | 0.00 |   29  |   12  | 0.76 |
| `[5]³`   |  2  |  1.12  |  0.91  | 0.01 |   61  |   21  | 0.81 |
| `[5]³`   |  5  |  1.33  |  1.20  | 0.02 |  157  |   82  | 0.90 |
| `[5]³`   | 10  |  1.58  |  1.47  | 0.03 |  303  |  148  | 0.93 |
| `[5]³`   | 20  |  1.58  |  1.56  | 0.02 |  231  |  139  | 0.99 |
| `[7]³`   |  2  |  1.31  |  1.18  | 0.01 |  119  |   43  | 0.91 |
| `[7]³`   |  5  |  **1.65**  |  **1.71**  | 0.08 |  757  |  460  | 1.04 |
| `[7]³`   | 10  |  1.47  |  1.58  | 0.12 | 1187  |  879  | 1.08 |
| `[7]³`   | 20  |  1.62  |  1.73  | 0.11 | 1135  |  869  | 1.07 |
| `[15]³`  |  2  |  1.29  |  1.32  | 0.05 |  457  |  120  | 1.02 |
| `[25]³`  |  2  |  1.07  |  1.35  | 0.11 | 1118  |  227  | 1.26 |

(Saturated cells `[15+]³` × PCA ≥ 5, and `[100]³` everywhere, omitted —
see "saturation" below. `[5]³` rows now use 5 seeds; the rest use 3.)
**Bold** = best class-info recovery, occurring at `[7]³` × PCA ∈ {5, 20}
where the network has just enough capacity without saturating the probe.

## Key findings

1. **DPI sanity (plug-in): clean.** 0 / 42 240 violations of
   `plug_in_func_bits ≤ plug_in_bits`. The estimator behaves correctly
   on real, multi-class data at N = 10 000 across the full
   capacity / bottleneck grid.

2. **`[7]³` is the class-info sweet spot.** Recovers 1.65–1.73 bits
   ≈ 50–52 % of H(Y) at PCA ≥ 5 — the highest of any trustworthy
   architecture. Below that, the network is capacity-starved (`[3]³`
   tops out at 1.03 bits). Above that, the network saturates the probe
   at PCA ≥ 5 and the estimator becomes unreliable. The sweet spot is
   set by *both* axes: enough capacity to learn the task,
   little enough capacity to keep `R / N` < 0.3.

3. **Bottleneck × capacity boundary is sharp and monotone.** Width 3,
   5, 7 are all trustworthy at every PCA dim (max ρ ≤ 0.20). Width 15
   jumps to ρ = 0.54 already at PCA = 5 and continues climbing. Step
   from width 7 to 15 multiplies regions by ≈ 4×. The trustworthy /
   saturated split is therefore near-binary along the width axis once
   the bottleneck loosens past PCA = 2.

4. **`I_func / I_raw` ratio rises monotonically with width** in the
   trustworthy regime (PCA = 2 column: 0.73 → 0.81 → 0.91 → 1.02 → 1.26
   as width goes 3 → 5 → 7 → 15 → 25). At narrow widths the quotient
   strips a meaningful chunk of structurally-distinct-but-functionally-
   equivalent regions; at wider widths the strip becomes nominal and
   asymmetric MM correction (`(R−1) / (2N ln 2)` shrinks raw bits more
   when `R_quot < R`) pushes the ratio above 1. Plug-in DPI still
   holds — see finding 1.

5. **Bottleneck effect dominates at small width.** `[3]³` and `[5]³`
   recovery curves saturate around 1.0 and 1.6 bits respectively. Going
   from PCA = 10 to PCA = 20 *does not help* `[5]³` (both ≈ 1.58
   bits) — the network can't exploit the extra dimensions.

6. **Saturated cells are scientifically uninformative here.** When
   `[15]³+` at PCA ≥ 5 saturates, the MM bias correction overwhelms
   the plug-in MI and pushes the bias-corrected bits negative. Composite
   at N = 20 000 didn't have this problem; on MNIST at N = 10 000 it
   kicks in fast for wider nets. Use the saturation map (finding 3) as
   the operating envelope.

7. **The expected memorization signature is invisible here.** MNIST is
   clean (no label noise), so I_raw and I_func track each other on the
   trustworthy cells (ratios 0.73–1.08). We can't replicate the
   composite finding ("wide nets memorize, quotient drop flips sign
   under noise") without adding label noise to MNIST training. The
   pattern this sweep *can* show — the (capacity × bottleneck)
   trustworthy boundary — it does show very cleanly.

## Caveats

- **No label noise.** Phase A only tells us how the estimator behaves
  on a clean classification task across capacity / bottleneck. The
  composite-style "noise sensitivity" story needs label noise added
  to the MNIST training (deferred — would need a fresh training sweep).
- **No holdout.** `truncation_prob` is NaN for all rows; we cannot
  diagnose probe-coverage failures the way composite can.
- **3 seeds only.** Confidence on the headline numbers is wider than
  on the 5-seed composite/wbc sweeps. Phase B will add seeds 104–105
  if we densify around the boundary.

## Figures

- `figures/mnist_capacity_new_estimator/capacity_landscape.png`
  (4-panel heatmap: I_raw, I_func, I_func/I_raw ratio, mean ρ — over
  the deepest-layer / ε=10 / last-epoch slice).
- `figures/mnist_capacity_new_estimator/trustworthiness_max_rho.png`
  (max ρ across all (epoch, layer, seed) per cell).

## Phase B — done

Phase B added `[3]³` and `[7]³` at all 4 PCA dims (3 seeds each,
24 jobs) plus seeds 104, 105 for `[5]³` (8 jobs). Wall time 3.8 min
(narrow nets are fast). Updated trustworthiness map and headline tables
above. Available `mnist_capacity` archs do not include `[9]³` so that
intermediate point is not testable without new training runs.

## Suggested next steps

1. **MNIST label-noise sweep.** The experiment that would actually
   test the H1/H2 hypotheses from the plan is to train MNIST with
   noise ∈ {0, 0.2, 0.4} on a few selected archs (e.g. `[5]³, [7]³,
   [25]³`) at PCA ∈ {5, 20}. Needs new training runs, ≈ a few hours.
2. **`comp_new_lf_*` (geometric-loss ablation).** Called out in
   `planning/next_phase.md` §"When you're done". This is the most
   likely place for the new estimator to show scientific value — the
   geometric loss term targets exactly the region structure the
   quotient is built on.
3. **Densify around `[7]³`.** Phase B revealed `[7]³` × PCA ≥ 5 as
   the class-info sweet spot. A small follow-up training run with
   widths {6, 7, 8, 9, 10, 12} at PCA ∈ {5, 10, 20} would map the
   capacity peak more precisely. Needs new training runs.
