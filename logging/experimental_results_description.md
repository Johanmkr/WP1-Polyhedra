# Experimental Results Description

This file describes the concrete results from the six active figures for use
in writing the experimental section of the paper. Figures are listed in
`current_figures.md`. Results are organised around the two main claims.

Active figures: pedagogical_figure1, calibration_scatter_raw,
layer_profile_last_epoch, mnist_capacity_bars_per_arch, mnist_rho_vs_eps,
rho_func_layerwise.

---

## Claim 1 — The routing estimator is calibrated against established MI baselines

### Setup
- Datasets: Composite (7-class synthetic, N=10k), WBC (2-class, N=569),
  MNIST FC (10-class, PCA-10 inputs, N=10k test set)
- Architectures: 3-layer and 5-layer MLPs with widths 5, 9, 25 (Composite/WBC)
  and widths 3, 5, 7 (MNIST FC)
- Estimator: plug-in routing MI (no bias correction)
- Operating point: last training epoch (150), deepest hidden layer
- Baselines: histogram binning (K=8), k-means (K=|Y|), KSG (k=3)

### Calibration scatter (Figure: calibration_scatter_raw)
- The plug-in routing estimator correlates strongly with all three baselines
  across all datasets and architectures.
- Points cluster tightly around the y=x diagonal in all three panels,
  with Pearson r values close to 1.
- The agreement holds across very different datasets (binary WBC vs
  10-class MNIST) and architecture sizes, demonstrating the estimator is
  not tuned to any particular setting.

### Layerwise profile (Figure: layer_profile_last_epoch)
- At every hidden layer (1 through 5 for Composite/WBC, 1 through 3 for MNIST),
  plug-in routing MI tracks the three baselines throughout the network depth.
- The routing estimator is consistently slightly below KSG, which is expected:
  KSG uses the continuous input representation while routing MI is bounded by
  the discrete partition entropy H(Ω).
- The gap between routing MI and KSG narrows in deeper layers where the
  partition becomes more class-discriminative.

---

## Claim 2 — Functional equivalence extends the estimator to high-dimensional settings

### Setup
- Dataset: MNIST with PCA-reduced inputs at dimensions d ∈ {2, 3, 4, 5, 10, 15, 20}
- Architectures: 3-layer MLPs with widths 7, 15, 25, 50
- Functional equivalence criterion: relative Frobenius distance on the active
  subnetwork matrix Ã, threshold ε ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0}
  (bounded in [0, 2], scale-invariant across depth)
- Operating point: last training epoch (150), last hidden layer (layer 3)

### The saturation problem (Figure: mnist_capacity_bars_per_arch)
- For small PCA dims (d ≤ 5), plug-in routing MI (I_raw) sits well below
  H(Y) = 3.319 bits, indicating an informative, non-saturated estimate.
- As d grows (d = 10, 15, 20), I_raw approaches H(Y) for wider architectures
  (width 25, 50), indicating the estimator has entered the fine-resolution
  regime where the partition is too fine to carry class information.
- This saturation is more severe for wider networks (more regions) and larger
  input dimensions (more unique routing paths).

### Functional equivalence as correction (Figure: mnist_capacity_bars_per_arch)
- I_func at small ε (0.1, 0.2) is nearly identical to I_raw: epsilon is too
  small to merge any regions, so the quotient partition equals the original.
- I_func at moderate ε (0.3–0.5) provides compression that recovers
  informative estimates for large d: bars drop meaningfully below I_raw
  and below H(Y), indicating genuine class structure is being measured.
- I_func at large ε (1.0, 2.0) over-compresses: too many regions merge and
  the estimate collapses toward lower values even for small d where
  saturation was not a problem.
- The useful ε range appears to be approximately 0.2–0.5, where compression
  is strong enough to escape saturation without destroying class structure.

### Compression ratio ρ_func (Figure: mnist_rho_vs_eps)
- At ε = 0, ρ_func = 1 for all (arch, PCA dim) combinations: no merging.
- For small d (d = 2, 3), ρ_func drops sharply even at small ε: many routing
  paths implement nearly identical linear operators (high reuse), confirming
  the network is not in the saturation regime.
- For large d (d = 15, 20), ρ_func stays close to 1 until ε ≈ 0.2–0.3,
  then drops: the network creates mostly unique linear operators per routing
  path (saturation regime), and only larger ε triggers merging.
- Wider architectures (width 50) show higher ρ_func at all ε compared to
  narrower ones (width 7), consistent with more regions and more unique operators.
- The ε range where ρ_func transitions (0.2–0.5) matches the range where
  I_func provides informative compression in the bar chart, confirming the
  two quantities measure the same underlying phenomenon.

### Layerwise compression profile (Figure: rho_func_layerwise)
- At ε = 0 (raw), ρ_func ≈ 1 at all layers — every routing path is unique.
- As ε increases, ρ_func drops across all layers, with the effect most
  pronounced at the deepest hidden layers where the network is most expressive.
- At ε = 1.0–2.0 most regions are merged and the quotient partition becomes
  coarse throughout the network.

### Summary of the functional equivalence argument
The raw routing estimator saturates at H(Y) when the input dimension is large
relative to the training set size (fine-resolution regime), because the network
assigns almost every sample to a unique routing path. The functional equivalence
quotient identifies routing paths that implement the same linear operator (up to
relative tolerance ε) and merges them, reducing the effective number of regions
and restoring sensitivity to class structure. The relative Frobenius criterion
is scale-invariant across network depth and bounded in [0, 2], making ε
interpretable: ε = 0 recovers the raw estimate; ε ≥ 1 merges most regions.
The useful operating range is ε ∈ [0.2, 0.5].
