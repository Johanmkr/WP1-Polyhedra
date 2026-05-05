# Claude Code Instructions: Routing Estimator Updates

> **Before starting:** There are several open questions at the bottom of this
> file (§ "Questions for the user"). Resolve them before executing any task
> that touches file paths, column names, or epsilon/PCA values.

---

## Task 1 — Replace the functional equivalence definition

### Motivation

The current definition checks `||A_omega - A_omega'|| < eps` **and**
`||c_omega - c_omega'|| < eps`. Both are wrong for disjoint-domain regions:

1. **Bias check is meaningless.** `c_omega` encodes the region's location in
   input space, not the character of its transformation. Two regions that
   implement the same linear operator will generically have different biases
   just because they sit in different parts of the input domain.

2. **Absolute epsilon is not scale-invariant.** `A^l_omega` is a product of
   `l` weight matrices whose norms compound with depth, so the same `eps`
   means something completely different at layer 2 vs layer 5.

### New definition

Two regions `omega` and `omega'` are functionally equivalent at layer `l` if:

```
||A^l_omega  -  A^l_omega'||_F
─────────────────────────────────  <  eps
0.5 * (||A^l_omega||_F + ||A^l_omega'||_F)
```

**The bias `c_omega` is dropped from the criterion entirely.**

### Implementation steps

1. **Find the equivalence function.** Search the codebase for the function
   that groups regions by functional equivalence. Look for:
   - calls to `np.linalg.norm` on matrix differences
   - comparisons involving both an `A` matrix and a `c` / `bias` vector
   - words like `functional`, `equiv`, `quotient`, `group`, `cluster`
   in file names and function names.

2. **Replace the comparison logic.** The current check probably looks
   something like one of these:

   ```python
   # Pattern A — direct norm comparison
   np.linalg.norm(A1 - A2) < eps and np.linalg.norm(c1 - c2) < eps

   # Pattern B — concatenated flattened vector
   np.linalg.norm(np.concatenate([A1.ravel(), c1]) -
                  np.concatenate([A2.ravel(), c2])) < eps

   # Pattern C — tolerance on each element
   np.allclose(A1, A2, atol=eps) and np.allclose(c1, c2, atol=eps)
   ```

   Replace **all** such patterns with:

   ```python
   def _relative_frob(A1: np.ndarray, A2: np.ndarray) -> float:
       """Relative Frobenius distance between two matrices."""
       denom = 0.5 * (np.linalg.norm(A1, "fro") + np.linalg.norm(A2, "fro"))
       if denom < 1e-12:          # both are essentially zero matrices
           return 0.0
       return np.linalg.norm(A1 - A2, "fro") / denom

   def are_functionally_equivalent(A1: np.ndarray,
                                   A2: np.ndarray,
                                   eps: float) -> bool:
       """True if A1 and A2 implement the same linear transformation
       up to relative tolerance eps.  Bias is excluded by design."""
       return _relative_frob(A1, A2) < eps
   ```

3. **Remove all bias arguments** from this function and from every call site
   that previously passed `c_omega`. If downstream code still needs `c_omega`
   for other purposes, leave it untouched — only remove it from the
   equivalence check.

4. **Update the active-subnetwork shortcut** (if present). The paper computes
   functional equivalence via the active sub-network matrix
   `A_tilde^l_omega = prod W^i[S^omega_i, S^omega_{i-1}]` rather than the
   full `A^l_omega`. Apply the same relative-Frobenius criterion to
   `A_tilde` matrices. The bias shortcut `c_tilde` should likewise be
   removed from the comparison.

5. **Re-run / invalidate any cached groupings.** If functional equivalence
   results are cached to disk (pickle, numpy, parquet), delete or regenerate
   them so downstream analysis uses the new definition.

### Sanity checks after the change

- `rho_func = |Omega_func| / |Omega|` should generally **decrease** relative
  to the old definition (more regions merge → more compression) because the
  strict bias check has been dropped.
- The epsilon plateau (the value of `eps` beyond which `|Omega_func|` stops
  shrinking) may shift; update `app:eps-plateau` figures if they are
  generated automatically.
- Verify the chain `tilde_I_func <= tilde_I_raw <= H(Y)` still holds
  numerically for all networks and layers.

---

## Task 2 — Simplify the `layer_profile_last_epoch` plot

### Goal

The plot currently shows four routing variants and three baselines (seven
lines per panel). Reduce it to **one routing estimator + three baselines**
for clarity.

### Lines to keep

| Line | Variable name (likely) | Style suggestion |
|------|------------------------|-----------------|
| Raw Miller–Madow corrected | `mi_raw_mm` / `tilde_I_raw` | solid, dark colour, circle markers |
| Binning K=8 | `mi_binning` | dashed, gray |
| k-means K=\|Y\| | `mi_kmeans` | dashed, olive |
| KSG k=3 | `mi_ksg` | dashed, cyan |

### Lines to remove

- Raw plug-in (`mi_raw_plugin` / `hat_I_raw`)
- Functional plug-in (`mi_func_plugin` / `hat_I_func`)
- Functional MM (`mi_func_mm` / `tilde_I_func`)

### Steps

1. Open `scripts/plot_layer_profile_last_epoch.py` (confirm path — see
   Questions below).

2. Find the list or loop that defines which estimators are plotted. It likely
   looks like one of:

   ```python
   estimators = ["mi_raw_plugin", "mi_raw_mm", "mi_func_plugin", "mi_func_mm"]
   # or
   for col in ["mi_raw_plugin", "mi_raw_mm", "mi_func_plugin", "mi_func_mm"]:
   ```

3. Reduce it to:

   ```python
   routing_estimators = ["mi_raw_mm"]   # only raw MM
   baseline_estimators = ["mi_binning", "mi_kmeans", "mi_ksg"]
   ```

4. Update legend labels, line colours, and any shared-axis limits.

5. The three-panel layout (Composite | WBC | MNIST FC) is unchanged.

6. Save to `figures/layer_profile_last_epoch.png` and `.pdf`.

---

## Task 3 — New plot: Functional MI vs epoch, sweeping PCA dimension and epsilon (MNIST capacity)

### Purpose

Show whether the new functional equivalence definition (relative Frobenius,
no bias) keeps the estimator informative as the MNIST input dimension grows.
Each panel fixes one epsilon value; lines within a panel show different PCA
dimensions coloured by a sequential palette.

### Data

Load all CSV files from `outputs/mnist_capacity/`.

**Assumed columns** (verify against actual headers before proceeding —
see Questions):

| Column | Description |
|--------|-------------|
| `epoch` | Training epoch |
| `layer` | Hidden layer index |
| `pca_dim` | Number of PCA components used as input |
| `epsilon` | Functional equivalence tolerance |
| `mi_func_mm` | Functional MI, Miller–Madow corrected |
| `seed` | Random seed |
| `architecture` or `width` | Network width specification |

### Aggregation

```python
import pandas as pd
import glob

df = pd.concat([pd.read_csv(f) for f in glob.glob("outputs/mnist_capacity/*.csv")])

# Keep only the deepest hidden layer
last_layer = df["layer"].max()   # or confirm this is the right selector
df = df[df["layer"] == last_layer]

# Mean and std across seeds (and architectures if pooling them)
agg = (df.groupby(["epsilon", "pca_dim", "epoch"])["mi_func_mm"]
         .agg(["mean", "std"])
         .reset_index())
```

### Plot specification

```
Layout : 1 row × N_epsilon panels  (or 2 rows if N_epsilon > 4)
Figure size : (4 * N_epsilon, 4) inches

Within each panel:
  x-axis : epoch  (0 … 150)
  y-axis : tilde_I_func (bits), range [0, H(Y)+0.1]
  Lines  : one per pca_dim value, coloured by viridis sequential map
           (darker = smaller pca_dim, lighter = larger pca_dim)
  Band   : ±1σ shaded, alpha=0.15
  H(Y)   : horizontal dashed line at 3.319 bits, labelled "H(Y)"
  Title  : "ε = {epsilon_value}"
  Legend : pca_dim values, placed in top-left or outside right

x-axis label : "Epoch"
y-axis label : "$\\tilde{I}_{\\mathrm{func}}$ (bits)"

Colourbar (optional) : show pca_dim values on a shared colourbar to the
                        right of the figure instead of a per-panel legend
```

**Key visual question**: For which epsilon values does `tilde_I_func` remain
above zero and below `H(Y)` across all PCA dimensions throughout training?
Estimators that collapse to zero (support too fine) or saturate at `H(Y)`
immediately (support too coarse / epsilon too large) indicate the boundary
of the useful regime.

### Save

```
figures/mnist_functional_pca_sweep.png
figures/mnist_functional_pca_sweep.pdf
```

---

## Task 4 — Suggested experiments

The following experiments are designed to back up the theoretical claims
using data already available in the `capacity/` and `label_noise/` folders.
Each is a new plotting script. Implement them in order of priority.

---

### Experiment A — Label noise vs routing information (raw MM)
**Priority: High**

**Theory.** Noisy labels mix class membership within regions, reducing
`I(Y; Pi)`. The raw MM estimator should decrease monotonically with label
noise rate η, reaching ~0 at η = 1 (fully randomised labels).

**Data.** `outputs/label_noise/` — confirm it contains a column for noise
rate (likely `eta` or `label_noise`) and `mi_raw_mm`.

**Plot.**
- x-axis: η (noise rate, 0 … 1)
- y-axis: `tilde_I_raw` at last layer, last epoch (mean ± 1σ across seeds)
- One line per dataset (Composite, WBC, MNIST FC)
- Normalise y-axis by `H(Y)` per dataset so all three lines share [0, 1]
- Expected shape: monotone decrease, concave or roughly linear

**Theoretical anchor.** At η = 0 the value should match the
`layer_profile_last_epoch` deepest-layer value. At η = 1 the class
labels are independent of the input so `I(Y; Pi) → 0`.

**Script.** `scripts/plot_label_noise_mi_curve.py`

---

### Experiment B — Compression ratio ρ_func vs label noise
**Priority: High**

**Theory.** Under label noise the network does not need to learn
structured routing, so fewer functionally distinct regions are required.
`rho_func = |Omega_func| / |Omega|` should *increase* toward 1 as η
increases (less compression = less structured computation).

**Data.** `outputs/label_noise/` — needs `n_regions` (= `|Omega|`) and
`n_regions_func` (= `|Omega_func|`) or equivalently both MI estimates
(to infer the ratio from the bias correction term).

**Plot.**
- x-axis: η
- y-axis: `rho_func` at last layer, last epoch
- One line per dataset / architecture width
- Expected shape: monotone increase, possibly convex at high η

**Script.** `scripts/plot_label_noise_rho_func.py`

---

### Experiment C — Architecture width vs raw MM saturation
**Priority: Medium**

**Theory.** Wider networks enter the fine-resolution regime
(`|Omega_D| ≈ N`) earlier. The raw MM estimator should plateau or
become uninformative above a threshold width, while the functional
quotient should extend the useful range.

**Data.** `outputs/capacity/` — varies width (confirm column name).

**Plot.**
- x-axis: architecture width (neurons per layer)
- Left y-axis: `tilde_I_raw` at last layer, last epoch (mean ± 1σ)
- Right y-axis: `rho_func` at same layer / epoch
- One set of curves per dataset
- Mark (vertical dashed line) the width at which `|Omega_D|` first
  exceeds `N / 10` as a "fine-resolution warning" threshold

**Script.** `scripts/plot_capacity_width_sweep.py`

---

### Experiment D — Functional MI vs raw MM on MNIST: does the new definition close the gap?
**Priority: Medium**

**Theory.** After dropping the bias from the equivalence criterion,
`tilde_I_func` should be **≥** the old functional estimate (more merging
→ higher lower bound). The gap between `tilde_I_func` and `tilde_I_raw`
quantifies how much compression the new definition provides. The KSG
gap should remain, confirming it is genuinely due to `I(Y; T | Pi)`,
not to the equivalence definition.

**Data.** `outputs/mnist_capacity/` — at η = 0, last epoch, last layer.

**Plot.**
- x-axis: PCA dimension
- y-axis: MI in bits
- Three lines: `tilde_I_raw`, `tilde_I_func` (new definition, optimal ε),
  KSG baseline
- Expected: raw and functional track each other closely; both sit
  consistently below KSG

**Script.** `scripts/plot_mnist_functional_vs_ksg.py`

---

### Experiment E — ρ_func layerwise profile under label noise
**Priority: Low**

**Theory.** The universal V-shape in `rho_func` (=1 at layer 1, sharp
drop at layer 2, partial recovery at deeper layers) should flatten under
label noise. If the network does not need structured routing, the layer-2
trough should be shallower and deeper-layer recovery weaker.

**Data.** `outputs/label_noise/` — multiple η values, all layers.

**Plot.** Same 3-panel layout as `fig:rho-func` in the paper.
Within each panel, one line per η value (η = 0, 0.2, 0.5 suggested),
±1σ shading. The V-shape for η = 0 should match the existing figure.

**Script.** `scripts/plot_label_noise_rho_func_layerwise.py`

---

## General conventions for all new scripts

- Use `matplotlib` with the same style sheet as existing figures (check for
  a `plt.style.use(...)` call or a shared `plot_utils.py` / `style.py`).
- Save every figure as both `.png` (300 dpi) and `.pdf` to `figures/`.
- Aggregate across seeds with `groupby + agg(['mean', 'std'])`.
- Reference H(Y) values:
  - Composite: **2.640 bits**
  - WBC: **0.953 bits**
  - MNIST FC: **3.319 bits**
- Do not hard-code file paths; use `pathlib.Path` and a single
  `RESULTS_DIR` / `FIGURES_DIR` variable at the top of each script.

---

## Questions for the user

Please answer these before executing the tasks above.

1. **Functional equivalence location.** What is the file path and function
   name where functional equivalence is currently computed? Any pointer to
   where `A_omega` and `c_omega` matrices are compared would help.

2. **Layer profile script.** Is the plot script at
   `scripts/plot_layer_profile_last_epoch.py`? If not, where is it?

3. **CSV column names in `outputs/mnist_capacity/`.** Specifically:
   - What is the column for PCA dimension? (`pca_dim`? `n_pca`? `input_dim`?)
   - What is the column for epsilon? (`epsilon`? `eps`? `tol`?)
   - What is the functional MI column? (`mi_func_mm`? `I_func`?)
   - Is there a `layer` column and does it index from 0 or 1?

4. **Available epsilon and PCA values.** What are the epsilon values present
   in the mnist_capacity data? What PCA dimensions are tested?

5. **Column names in `outputs/label_noise/`.** What is the noise-rate column
   called? (`eta`? `label_noise`? `noise_rate`?) Is `rho_func` stored
   directly, or must it be computed from `n_regions` and `n_regions_func`?

6. **Column names in `outputs/capacity/`.** What does the width/depth axis
   look like? Is it a single `width` column or an `architecture` string?

7. **Caching.** Are functional equivalence groupings cached to disk anywhere?
   If so, what format and where?
