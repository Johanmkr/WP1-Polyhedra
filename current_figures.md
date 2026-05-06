# Current Figures

Log of active plotting scripts, their outputs, and narrative purpose.
Updated as scripts are revised.

---

## Narrative

The figures are organised around two claims:

**Claim 1 — The routing estimator is calibrated against established MI baselines.**
Supported by the calibration scatter and the layerwise profile.

**Claim 2 — Functional equivalence (ε-quotient) extends the estimator to
high-capacity / high-dimensional settings where raw routing MI saturates.**
Supported by the capacity bars and the ρ_func vs ε plot.

---

## Active plots

### `scripts/plot_figure1_pedagogy.py`
**Output:** `pedagogical_figure1.png/.pdf` ✓

Figure 1: illustrates how ReLU networks partition input space via activation
patterns. Trains a small 2→4→4→2 network on the moons dataset and shows:
(a) after layer 1 — hyperplanes creating convex regions, (b) after layers
1+2 — piecewise-linear refinement, (c) network diagram with one region ω
identified by its activation pattern π_ω. Run with `--retrain` to force
retraining.

**Purpose:** Introductory figure explaining the geometric partition and the
activation-pattern encoding underlying the routing estimator.

### `scripts/plot_calibration_scatter.py`
**Output:** `calibration_scatter_raw.png/.pdf` ✓

Scatter: plug-in routing MI (`plug_in_bits`, x-axis) vs each baseline
(binning K=8, k-means K=|Y|, KSG k=3, y-axis). One panel per baseline,
shared y-axis. Points coloured by dataset (Composite, WBC, MNIST FC PCA-10).
Diagonal y=x and Pearson r annotated. Last epoch, deepest layer.

**Purpose:** Establishes that routing MI agrees with established methods
across all conditions.

### `scripts/plot_layer_profile_last_epoch.py`
**Output:** `layer_profile_last_epoch.png/.pdf` ✓

Plug-in routing MI (`plug_in_bits`) as a function of layer depth at last
epoch (η=0, ε=1.0), averaged across seeds and architectures. Three panels:
Composite (5 layers), WBC (5 layers), MNIST FC PCA-10 (3 layers). Three
baselines overlaid (binning, k-means, KSG).

**Purpose:** Shows calibration holds across all layers, not just the deepest.

### `scripts/plot_mnist_capacity_bars.py`
**Output:** `mnist_capacity_bars_per_arch.png/.pdf` ✓

Grouped bar chart: plug-in I_raw and I_func vs PCA dimension at last epoch,
last hidden layer. One bar group per PCA dim: I_raw (dark gray) and I_func
at ε = 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0 (sequential Blues). Architectures:
widths 7, 15, 25, 50. 2×2 grid (14×5.5 in), shared axes. H(Y) reference.

**Purpose:** Shows I_raw saturates toward H(Y) for large PCA dims; I_func
with moderate ε recovers an informative estimate.

### `scripts/plot_mnist_functional_pca_sweep.py --type rho`
**Output:** `mnist_rho_vs_eps.png/.pdf` ✓

ρ_func vs ε at last epoch, last hidden layer. 2×2 grid of architecture panels
(widths 7, 15, 25, 50). Lines coloured by PCA dim (tab10). Shared axes,
wide format (14×7 in).

**Purpose:** Shows the compression ε provides and where functional merging
begins. Justifies the ε range used in I_func.

### `scripts/plot_rho_func_layerwise.py`
**Output:** `rho_func_layerwise.png/.pdf` ✓

ρ_func by layer depth for multiple ε values (0.0, 0.1, 0.3, 0.5, 1.0, 2.0),
averaged over architectures and seeds. Three panels: Composite, WBC, MNIST.
Shared y-axis.

**Purpose:** Shows how functional compression varies across network depth
and how it responds to different ε values.

---

## Notes

- All figures saved to `/home/johan/Documents/phd/WP1/neurips_2026_paper/figures/`
  via `savefig()` from `src_experiment/utils.py`.
- Functional equivalence uses relative Frobenius criterion (bias excluded).
  Default ε=1.0 for all "at a fixed epsilon" plots.
- MNIST FC baselines (`mnist_fc_baselines.csv`) only cover target_dim=10.
