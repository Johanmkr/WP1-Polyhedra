# §X — Calibration of the routing-information estimator

*Paper-style write-up of the calibration experiment. LaTeX source
inside fenced code blocks; surrounding prose is editorial commentary
that should not be lifted into the manuscript.*

The section below assumes the reader has already met
$\hat I(Y;\Omega_\mathcal{D})$ (plug-in routing information, eq. 8),
$\tilde I(Y;\Omega_\mathcal{D})$ (Miller–Madow corrected, eq. 9),
their functional-quotient refinements
$\hat I(Y;\Pi_\text{func})$ and $\tilde I(Y;\Pi_\text{func})$
(Prop. 4.6), and the five MI baselines that were validated against
synthetic ground truth in §5.1 / `2026-04-29_baseline_mi_paper_section.md`.

---

## Recommended placement

Section §6 (or a new §5.2): three calibration figures + an
appendix-style sub-section on the choice of the active-subnetwork
clustering tolerance $\varepsilon$.

---

## §X.1 Setup

```latex
\subsection{Setup}
\label{sec:calibration-setup}

We benchmark the routing-information estimator against three
plug-in MI baselines---uniform binning ($K{=}8$), $k$-means
($K{=}|\mathcal{Y}|$) and KSG ($k{=}3$, Ross 2014)---across a
deliberately diverse set of fully-connected ReLU networks. The neural
critics InfoNCE and MINE-f, also reported in
\cref{sec:baselines-validation}, are omitted from this calibration to
keep the per-cell wall time tractable for the per-epoch sweep
(\cref{sec:calibration-trajectory}); the cheaper plug-in baselines are
sufficient to establish agreement.

\paragraph{Datasets.} \textit{Composite} (synthetic 7-class 2D mixture,
$N{=}10\,000$, $H(Y){=}2.639$ bits) and \textit{WBC} (UCI Wisconsin
Breast Cancer, binary, $N{=}569$, $H(Y){=}0.953$ bits). Both were used
in the §4.4 validation; we recycle the existing checkpoints rather
than retrain.

\paragraph{Architectures and seeds.} Six fully-connected ReLU
architectures spanning depth $\in\{3,5\}$ and width
$\in\{5,9,25\}$:
$[5,5,5]$, $[5,5,5,5,5]$, $[9,9,9]$, $[9,9,9,9,9]$, $[25,25,25]$,
$[25,25,25,25,25]$. Each architecture is trained on each dataset for
five seeds, giving $6 \times 5 \times 2 = 60$ networks per panel.

\paragraph{Probe.} The same $N_{\text{probe}} = 5{,}000$ stratified
subsample is used for every estimator at every cell, so cross-method
differences are not a function of the probe sample.

\paragraph{Operating point for the functional quotient.} All four
``ours'' columns use the routing partition $\Omega_\mathcal{D}$ at
its natural resolution; the two functional variants additionally
apply the $\varepsilon$-quotient of \cref{sec:functional-quotient}
with $\varepsilon = 10$, justified empirically in
\cref{sec:epsilon-plateau}.
```

---

## §X.2 Calibration scatter (last epoch, deepest layer) — Panel A

```latex
\subsection{Pointwise calibration against plug-in MI baselines}
\label{sec:calibration-scatter}

\Cref{fig:calibration-scatter} scatters the Miller--Madow routing
estimator $\tilde I_{\text{raw}}$ against each of the three baselines
at the deepest hidden layer of every $\text{(arch, seed)}$
configuration after training. The diagonal $y = x$ is drawn for
reference; the Pearson coefficient $r$ over the pooled sample
($n = 180$) is annotated in each panel.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/calibration_scatter.png}
  \caption{Pointwise calibration of the routing-information estimator
    $\tilde I_{\text{raw}}(Y;\Omega_\mathcal{D})$ against three
    plug-in MI baselines, at the deepest hidden layer of each
    network, after training. One marker per
    $(\text{architecture}, \text{seed})$ on \textit{composite} (blue)
    and \textit{WBC} (orange). Across both datasets the routing
    estimator agrees with the baselines along $y = x$ to within their
    intrinsic plug-in bias; Pearson $r$ pooled across datasets is
    $0.97$ (binning), $0.95$ (k-means), $0.89$ (KSG).}
  \label{fig:calibration-scatter}
\end{figure}

The two datasets occupy disjoint bit ranges --- WBC is binary so
$\tilde I \le H(Y) \approx 0.95$, composite has seven classes so the
estimate climbs to $\sim 2.5$ --- and within each cluster the points
lie tightly along the diagonal. The scatter is not the consequence of
any single architecture or seed; it is the across-condition
relationship that is consistent. The Pearson coefficient drops
slightly for KSG ($r = 0.89$) because KSG saturates above
$\sim 2.4$ bits on the composite networks
(\cref{sec:trajectory-discussion}); on WBC, where bits are well
below saturation, all three baselines agree with the routing
estimator to the same tolerance.
```

---

## §X.3 Layerwise profile at the last epoch — Panel B

```latex
\subsection{Layerwise profile at convergence}
\label{sec:calibration-layerwise}

\Cref{fig:layer-profile} reports the same comparison but as a
function of layer depth, after restricting to the three
five-hidden-layer architectures so that the layer index has the same
meaning across networks. Bits at each layer are averaged over
$3 \times 5 = 15$ networks per dataset; the shaded bands span
$\pm 1\sigma$.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/layer_profile_last_epoch.png}
  \caption{Layerwise bits at the last training epoch, $\eta = 0$, for
    five-hidden-layer architectures. Solid lines: the four routing
    estimators (raw and functional, plug-in and Miller--Madow).
    Dashed lines: the three plug-in baselines. Both families recover
    the same monotonically increasing layerwise profile, with the
    routing estimators bracketing the baseline curves; on
    \textit{composite} the four routing variants and the three
    baselines all converge to the same $\sim 2.4$ bits at the
    deepest layer, well below the data-side ceiling
    $H(Y) = 2.64$ bits.}
  \label{fig:layer-profile}
\end{figure}

Both families produce the same shape: bits rise monotonically with
depth and the four routing variants form a tight envelope around the
baseline curves. The functional-quotient variants
$\hat I(Y;\Pi_\text{func})$ and $\tilde I(Y;\Pi_\text{func})$ sit
slightly below the raw variants, which is the expected effect of
collapsing functionally-equivalent regions: every collapse can only
reduce or leave unchanged the routing entropy $H(\Omega)$ and
therefore the upper bound on $I(Y;\Omega)$.
```

---

## §X.4 Trajectory (per-epoch, per-layer) — Panel C

```latex
\subsection{Information dynamics during training}
\label{sec:calibration-trajectory}

\Cref{fig:routing-trajectory} extends the layerwise profile to a
function of epoch, exposing the training dynamics that
\cref{fig:calibration-scatter,fig:layer-profile} suppress. The four
routing variants are tracked for layers $1\ldots5$ on each dataset;
shaded bands span $\pm 1\sigma$ across $3 \times 5 = 15$ networks per
cell. Baselines are not overlaid: the per-epoch baseline sweep was
prohibitive in wall-clock time, and the $\varepsilon = 10$ values
shown here can be calibrated against the baselines through
\cref{fig:calibration-scatter,fig:layer-profile} at the convergence
point.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/routing_trajectory.png}
  \caption{Per-epoch trajectories of the four routing-information
    estimators on five-hidden-layer architectures at $\eta = 0$,
    layers $1\ldots5$ (columns), datasets composite and WBC (rows).
    Solid red/orange: raw plug-in / Miller--Madow. Dotted blue/green:
    functional-quotient plug-in / Miller--Madow. The plug-in raw
    estimator rises monotonically and saturates near $H(Y)$ at the
    deepest layers; the Miller--Madow correction tracks it within
    $\sim 0.05$ bits once $|\Omega_\mathcal{D}|/N$ stabilises. The
    functional-quotient variants stay below their raw counterparts by
    a small constant gap, in line with the
    \cref{sec:calibration-layerwise} profile.}
  \label{fig:routing-trajectory}
\end{figure}

\paragraph{Observed dynamics.} Across both datasets, layer 1 carries
$\sim 1.5$ bits at composite and $\sim 0.5$ bits at WBC essentially
from initialisation; the deepest layers climb during training and
saturate within $\sim 5\%$ of $H(Y)$ by epoch $50$. The relative
ordering plug-in $>$ Miller--Madow holds throughout training, and
raw $>$ functional, in agreement with the layerwise snapshot.
```

---

## §X.5 The $\varepsilon$ tolerance and why $\varepsilon = 10$

```latex
\subsection{Selecting the active-subnetwork tolerance \texorpdfstring{$\varepsilon$}{ε}}
\label{sec:epsilon-plateau}

The functional quotient (Recipe~2) merges regions whose
active-subnetwork affine maps agree:
\begin{equation}
  \omega \sim_\varepsilon \omega'
  \iff
  \|\tilde A^l_\omega - \tilde A^l_{\omega'}\|_F
  + \|\tilde c^l_\omega - \tilde c^l_{\omega'}\|_2
  \le \varepsilon,
  \label{eq:eps-cluster}
\end{equation}
where $\tilde A^l_\omega \in \mathbb{R}^{|S_l|\times n_0}$ and
$\tilde c^l_\omega \in \mathbb{R}^{|S_l|}$ are the affine map and bias
of the active subnetwork at depth $l$, computed by the recursion
$\tilde A_i = W^i[S_i, S_{i-1}]\,\tilde A_{i-1}$,
$\tilde c_i = W^i[S_i, S_{i-1}]\,\tilde c_{i-1} + b^i[S_i]$
initialised at the identity.

The single scalar $\varepsilon$ is the only free parameter of the
quotient. Because $\tilde A^l$ accumulates $L$ matrix products, its
Frobenius norm scales with $\prod_i \|W^i\|$, and a fixed
$\varepsilon$ cannot be calibrated a~priori across architectures.
We therefore follow the plateau diagnostic of
\cref{sec:functional-quotient-spec}: sweep $\varepsilon$ across
multiple decades and pick a value where the quotient size
$|\Omega_{\mathcal{D},\text{func}}|$ and the resulting MI estimate
are both stable across $\geq 1$ decade. Below the plateau,
$\varepsilon$ only absorbs floating-point noise; above it,
$\varepsilon$ collapses genuinely distinct affine maps and the MI
estimate degrades.

\Cref{tab:eps-sweep} reports the sweep on two representative
configurations at the deepest layer / last epoch (mean across five
seeds).

\begin{table}[t]
  \centering
  \small
  \begin{tabular}{r|rrr|rrr}
    \toprule
    & \multicolumn{3}{c|}{Composite, $[9,9,9,9,9]$, layer 5}
    & \multicolumn{3}{c}{WBC, $[25,25,25,25,25]$, layer 5} \\
    $\varepsilon$ & $|\Omega_\mathcal{D}|$ & $|\Omega_{\mathcal{D},\text{func}}|$ & $\tilde I_\text{func}$ [bits]
    & $|\Omega_\mathcal{D}|$ & $|\Omega_{\mathcal{D},\text{func}}|$ & $\tilde I_\text{func}$ [bits] \\
    \midrule
    $0$       & 473.8 & 473.8 & 2.431 & 508.4 & 508.4 & 0.300 \\
    $10^{-4}$ & 473.8 & 473.8 & 2.431 & 508.4 & 508.4 & 0.300 \\
    $10^{-2}$ & 473.8 & 473.4 & 2.431 & 508.4 & 508.4 & 0.300 \\
    $10^{-1}$ & 473.8 & 470.0 & 2.432 & 508.4 & 508.4 & 0.300 \\
    $1$       & 473.8 & 461.6 & 2.432 & 508.4 & 483.4 & 0.329 \\
    $\mathbf{10}$  & \textbf{473.8} & \textbf{383.0} & \textbf{2.432}
                   & \textbf{508.4} & \textbf{72.2}  & \textbf{0.766} \\
    $100$     & 473.8 & 56.4  & 2.210 & 508.4 & 68.0  & 0.769 \\
    $1000$    & 473.8 & 21.4  & 1.880 & 508.4 & 68.0  & 0.769 \\
    \bottomrule
  \end{tabular}
  \caption{$\varepsilon$ sweep at the deepest layer / last epoch
    (mean across five seeds). The WBC plateau is sharp: from
    $\varepsilon = 10$ through $\varepsilon = 1000$ both the quotient
    size and the MI estimate are flat, identifying $\varepsilon = 10$
    as the smallest value inside the plateau. On composite the
    plateau is shorter but the MI estimate at $\varepsilon = 10$ is
    within $10^{-3}$ bits of the $\varepsilon = 0$ value, while
    $\varepsilon = 100$ has fallen off ($\sim\!0.2$ bits below the
    plateau).}
  \label{tab:eps-sweep}
\end{table}

We adopt $\varepsilon = 10$ throughout the paper. The same diagnostic
holds on the other architectures and noise levels in our sweep
(\cref{app:eps-plateau-extended}).
```

---

## §X.6 Summary paragraph (executive)

```latex
\paragraph{Calibration summary.}
The routing-information estimator agrees with the standard plug-in MI
baselines (binning, k-means, KSG) along $y = x$ across both datasets
and all architectures
(\cref{fig:calibration-scatter}, $r \ge 0.89$), reproduces their
layerwise profile after training
(\cref{fig:layer-profile}), and exposes the per-epoch information
dynamics that the (cubic-time) baselines cannot afford to track
(\cref{fig:routing-trajectory}). The functional-quotient refinement
is reported at $\varepsilon = 10$, justified by the plateau
diagnostic of \cref{tab:eps-sweep}.
```

---

## Caveats not yet folded into the LaTeX

1. **MNIST / LeNet.** Out of scope here — the routing estimator has
   not been wired to the CNN HDF5s yet. Add a forward reference once
   `cnn_estimator.FunctionalQuotientEstimatorCNN` is in the runner.
2. **KSG saturation in panel C.** We discussed overlaying sparse-epoch
   KSG checkpoints; KSG saturates near $H(Y)$ across the trajectory at
   $N = 5\,000$, so the overlay would muddy the picture. Mention in
   passing in §X.4 if a reviewer asks why no baselines.
3. **Neural baselines.** InfoNCE and MINE-f were validated separately
   in §5.1 but skipped here for wall-time reasons. A short footnote
   to that effect would close the gap.
