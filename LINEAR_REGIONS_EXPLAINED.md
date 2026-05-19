# How Linear Regions Are Found: A Technical Reference

This document explains the full pipeline from training a neural network to plotting its linear regions. It is intended as a reference for rebuilding the functionality in a new package, and notes explicitly where I am uncertain about the exact behavior.

---

## 1. Big-Picture Pipeline

A ReLU network partitions its input space into *linear regions* — polytopes where the same set of neurons is active, and the network's output is a single affine function. This codebase discovers and stores those regions in four steps:

```
Step 1 (Python)  Train network; snapshot weights at selected epochs → HDF5
Step 2 (Julia)   Read weights; discover regions; save topology back → HDF5
Step 3 (Python)  Read HDF5; estimate MI / entropy / KL metrics → plots
Step 4 (Python)  Read HDF5; draw region polytopes in 2D → plots
```

The whole pipeline is driven by `run_pipeline.sh`:

```bash
./run_pipeline.sh configs/my_experiment.yaml [NUM_THREADS]
```

Steps 3 and 4 can be run independently on any existing HDF5 file.

---

## 2. Configuration

Experiments are configured via YAML files (see `configs/template.yaml`). Key fields:

| Field | Meaning |
|---|---|
| `experiment_name` | Used to name the output directory and HDF5 file |
| `output_dir` | Root directory for outputs |
| `architecture` | List of hidden layer widths, e.g. `[8, 8]` |
| `global_seed` | Controls train/test split and data shuffle |
| `model_seed` | Controls weight initialization only |
| `epochs` | Total training epochs |
| `save_interval` | Save every N-th epoch |
| `save_epochs` | Also save these specific epochs (e.g., `[0, 1, 2, 5, 10]`) |

The output HDF5 file is written to:
```
<output_dir>/<experiment_name>/seed_<model_seed>.h5
```

---

## 3. HDF5 File Structure

The same HDF5 file is written by Python training and then extended in-place by Julia. Here is the full structure after both steps have run.

### 3.1 Written by Python (Step 1)

```
/metadata/                         (HDF5 group, attrs only)
    @attrs:
        experiment_name            string
        dataset                    string
        architecture               int array
        global_seed                int
        model_seed                 int
        epochs                     int
        inferred_input_size        int
        inferred_num_classes       int
        inferred_input_shape       int list
        ... (all other config keys)

/epochs/
    /epoch_0/                      (created at each save point)
        @attrs:
            train_loss             float
            test_loss              float
            train_accuracy         float
            test_accuracy          float
        l1.weight                  float32 array, shape (out_features, in_features)
        l1.bias                    float32 array, shape (out_features,)
        l2.weight                  ...
        l2.bias                    ...
        l3.weight                  ...   ← output layer included
        l3.bias                    ...
    /epoch_10/
        ...

/training_results/
    train_loss                     float array, shape (epochs,)
    test_loss                      float array, shape (epochs,)
    train_accuracy                 float array, shape (epochs,)
    test_accuracy                  float array, shape (epochs,)

/points                            float32 array, shape (N, input_dim)   ← test set inputs
/labels                            int array,     shape (N,)              ← test set labels
```

**Weight layout note.** PyTorch stores `nn.Linear` weights in row-major (C) order with shape `(out_features, in_features)`. When written to HDF5 with h5py this is a row-major array. Julia's HDF5.jl reads in column-major (Fortran) order, so the dimensions appear *transposed*: Julia sees shape `(in_features, out_features)`. Julia corrects for this with `permutedims(data, (2, 1))` (see `save_tree.jl:read_state_dict_from_h5`). Python's `reconstruction.py` does the same with `.T` when reading back. This is easy to get wrong when porting.

**Layer naming.** The keys (`l1.weight`, `l1.bias`, …) come from PyTorch's `state_dict()` using the layer names set in `NeuralNet._build_layers`. The output layer is stored under `l{len(hidden_sizes)+1}.weight`. Julia's `read_state_dict_from_h5` matches keys containing the substring `"weight"` or `"bias"`, so it picks up all layers including the output layer. *I am not certain whether including the output layer's weights in the tree construction is intended or incidental — see §5.4.*

`reconstruction.py` also has a conditional branch for a `model/` subgroup containing keys `W_0`, `b_0`, `W_1`, `b_1`, … but the current Python training code does **not** create that subgroup. It appears to be dead code from an older version.

### 3.2 Written by Julia (Step 2)

Julia writes the following datasets into each `/epochs/epoch_N/` group (in-place, same HDF5 file):

```
parent_ids        int32 array,   shape (num_nodes,)
layer_idxs        int8  array,   shape (num_nodes,)
volumes_ex        float64 array, shape (num_nodes,)   ← exact volume (0 if not computed)
volumes_es        float64 array, shape (num_nodes,)   ← estimated volume (0 if not computed)
bounded           bool  array,   shape (num_nodes,)
centroids         float64 array, shape (input_dim, num_nodes)  ← stored column-major

qlw_flat          int8  array,   shape (total_activation_bits,)
qlw_offsets       int64 array,   shape (num_nodes + 1,)

active_flat       int32 array,   shape (total_active_indices,)
active_offsets    int64 array,   shape (num_nodes + 1,)
```

The tree is serialized as a BFS-ordered flat array. `parent_ids[i]` is the index of node `i`'s parent (`-1` for the root). `layer_idxs[i]` is the depth of node `i` (0 = root, 1 = first hidden layer, …, L = deepest layer).

`qlw_flat` / `qlw_offsets` form a ragged array: the activation signature of node `i` is `qlw_flat[qlw_offsets[i] : qlw_offsets[i+1]]`. For the root (layer 0) this is empty. For a node at layer `l`, it is a bit vector of length equal to the number of neurons in that layer.

`active_flat` / `active_offsets` work the same way and store the indices of the *non-redundant* halfspace constraints for each region. This is only populated when using exact construction (`--exact_regions`); sparse construction leaves it empty.

`centroids` stores a feasible interior point for each region (a point that lies inside the polytope). For sparse construction this is just whatever training/test data point was routed through that region. For exact construction it is the result of an LP.

---

## 4. Julia Module Architecture (`geobin_jl/`)

The module is loaded as a single Julia module `Geobin` from `geobin_jl/geobin.jl`. All sub-files are included in dependency order:

```
geobin.jl          ← Module entry point, imports, re-exports
  region.jl        ← Region struct
  geometry.jl      ← Halfspace LP utilities
  utils.jl         ← find_hyperplanes, compute_path_geometry, volumes
  tree.jl          ← Tree struct, get_regions_at_layer, print_tree_summary
  construction.jl  ← Exact DFS construction
  verification.jl  ← Sanity checks
  pruning.jl       ← Volume/feasibility-based pruning
  save_tree.jl     ← HDF5 read/write
  sparse_construction.jl  ← Data-driven sparse construction
  volume.jl        ← Parallel volume computation (CDDLib, volesti)
```

### Dependencies

| Package | Role |
|---|---|
| `HDF5.jl` | Read/write experiment files |
| `JuMP` + `HiGHS` | LP feasibility checks, active-index detection |
| `Polyhedra` + `CDDLib` | H-rep / V-rep conversion, vertex enumeration, exact volume |
| `ProgressMeter` | Progress bars |
| `RCall` + `volesti` | Monte Carlo volume estimation (optional) |
| `Distributed` | Parallelism for exact volume computation |

---

## 5. Core Algorithm: Finding Linear Regions

### 5.1 What a Region Is

A linear region of a ReLU network is defined by a binary *activation signature* `q` — for each layer, a bit per neuron saying whether that neuron is active (1) or inactive (0). All points in a region share the same `q`. The region is the set of input points `x` consistent with that pattern.

In the `Region` struct (`region.jl`), this is stored as a `BitVector qlw`. Deeper layers have longer signatures (concatenation of all layer signatures up to that depth), but in practice each `Region` node only stores the signature for *its own layer*; the full path is reconstructed by tracing up through `parent` pointers.

### 5.2 Deriving Halfspace Constraints

The key mathematical operation is converting an activation path into a system of linear inequalities `D * x ≤ g`. This is implemented in `compute_path_geometry` (`utils.jl`) and mirrored identically in `Tree.get_path_inequalities` (`geobin_py/reconstruction.py`).

The derivation proceeds layer by layer, maintaining an *accumulated affine map* `(A, c)` such that the pre-activation at layer `l` is `W_l * A * x + (W_l * c + b_l)`. At the root, `A = I`, `c = 0`.

For each layer `l` with weight `W`, bias `b`, and target activation pattern `q`:

```
W_hat  = W  * A_curr        # Effective weight matrix from input to this layer
b_hat  = W  * c_curr + b    # Effective bias

s      = -2*q + 1           # Sign flip: q=1 (active) → s=-1; q=0 (inactive) → s=+1
D_local = s .* W_hat        # Row-wise multiply
g_local = -(s .* b_hat)     # Element-wise

A_next  = q .* W_hat        # For active neurons only
c_next  = q .* b_hat
```

The condition encoded by `(D_local, g_local)` is that neuron `i` at layer `l` has activation `q[i]`. Specifically:
- Active neuron `i` (`q[i]=1`, `s[i]=-1`): pre-activation `> 0`, i.e., `-W_hat[i,:] * x ≤ b_hat[i]`
- Inactive neuron `i` (`q[i]=0`, `s[i]=+1`): pre-activation `≤ 0`, i.e., `W_hat[i,:] * x ≤ -b_hat[i]`

The full constraint set for a leaf region is the vertical stack of all `(D_local, g_local)` from layer 1 to `L`, giving a polytope in the original input space.

### 5.3 Exact Construction (`construct_tree!` in `construction.jl`)

Exact construction finds **all** non-empty regions using a DFS + facet-flipping strategy:

1. Start at the root. Evaluate the network at the root's feasible point to get an initial activation pattern `q_start`.
2. Check feasibility of `q_start` using an LP (`get_feasible_point` in `geometry.jl`). If feasible, create a child `Region`, recurse deeper.
3. Find which constraints in `D_local` are **non-redundant** (active constraints) by calling `find_active_indices_exact` or `find_active_indices_lp`. These active constraints are the *facets* of the current polytope.
4. For each active constraint index, flip the corresponding bit of `q_curr` to get a neighbor activation pattern. If not yet visited, push it onto the BFS queue.
5. This guarantees that every pair of adjacent regions (sharing a facet) is discovered.

**`find_active_indices_exact`** (`geometry.jl`): Uses CDDLib to compute the full H-representation → V-representation (vertex enumeration). This is mathematically clean but slow; it removes redundant halfspaces and returns the indices of non-redundant ones. It also determines boundedness (via checking for rays in the V-rep) and computes exact volume.

**`find_active_indices_lp`** (`geometry.jl`): Uses HiGHS LP to check redundancy by temporarily removing each constraint and checking whether the polytope expands. Faster for large constraint sets. Also deduplicates parallel constraints before running the LP.

The tree topology is layered: `Region.layer_number` matches the Julia layer index (1-indexed layers, 0 = root). A path from root to a leaf at depth `L` gives the complete activation path of a region.

### 5.4 Output Layer Inclusion

*I am uncertain about this.* Julia's `read_state_dict_from_h5` reads all weight/bias keys from the HDF5, including the output layer's linear transformation. This means `Tree.L` equals the total number of linear layers (hidden + output), not just the number of hidden ReLU layers. At the final depth, the "activation pattern" would be based on whether each output logit is positive. This may be intentional (finding class-decision regions in the last layer), or it may be a side-effect that doesn't cause problems because the output layer regions are not used downstream. Worth verifying when porting.

### 5.5 Sparse Construction (`construct_tree_sparse!` in `sparse_construction.jl`)

Sparse construction is the **default** mode (used unless `--exact_regions` is passed). It is much faster and memory-efficient:

1. For each data point (test set), compute its activation path through the network by running a forward pass: `q_l = (W_l * a + b_l > 0)`, then `a = q_l .* (W_l * a + b_l)` (the ReLU).
2. This is done in parallel across all threads for all data points.
3. Then the paths are inserted into the tree sequentially. A locking mechanism (`ReentrantLock`) protects the root node's children list during parallel insertion.
4. A `Region` is created only for each *distinct activation path* seen in the data. The stored `x` for each region is the first data point that was routed there.

**Key differences from exact construction:**
- Only regions that contain at least one data point are discovered.
- `active_indices` is **not** populated (empty `Int32[]`).
- Volume is **not** computed.
- There is no facet-flipping; neighbors are not explored.

The downstream Python code (`perform_number_count` in `reconstruction.py`) uses sparse-tree-style routing, which matches how the tree was built: it runs a forward pass to get `Q_path` and then routes points to the matching child at each level.

---

## 6. Volume Computation

Volume computation is optional and expensive. There are two modes:

**Exact volumes** (`compute_volumes_parallel!` in `volume.jl`): Uses `Distributed.pmap` to spread CDDLib vertex enumeration across worker processes (CDDLib's C library is not thread-safe). For each region, traces the activation path, reconstructs the full constraint set, and calls `Polyhedra.volume`. CDDLib is only safe with distributed (`-p N`) parallelism, not with `Threads.@threads`.

**Estimated volumes** (`estimate_volumes_parallel!` in `volume.jl`): Uses `volesti` (an R library, via `RCall`). Applies a hypercube bounding box to make regions finite, then calls `volesti::volume` on each bounded H-polytope. Uses a JuMP model pool for thread-safe recession cone checks. This requires R with the `volesti` package installed.

Both modes store results in `Region.volume_ex` and `Region.volume_es` respectively, which map to the `volumes_ex` / `volumes_es` HDF5 datasets.

---

## 7. Saving and Loading

### 7.1 Julia → HDF5 (`save_tree.jl`)

The tree is serialized by BFS traversal. Each node gets an integer index (its position in the BFS order, 0-indexed). `parent_ids[i]` stores the index of node `i`'s parent, enabling reconstruction of the tree structure without storing pointers.

Variable-length arrays (`qlw`, `active_indices`) are stored as flat ragged arrays using offset arrays (CSR-style), since HDF5 doesn't natively support ragged arrays.

`centroids` is saved transposed: Python shape is `(N, input_dim)` but Julia writes it as `(input_dim, N)` in column-major storage. The Python reader handles this (the comment in `reconstruction.py:163` says "Centroids are (N, D) in the fixed Julia script" — this matches after reading with h5py).

### 7.2 Python ← HDF5 (`geobin_py/reconstruction.py`)

`Tree._load_and_construct()` reads the flat arrays and rebuilds the Python object graph:
1. Reads all weight/bias datasets and stores them as `self.W` / `self.b` (with `.T` transpose).
2. Reads the flat topology arrays (`parent_ids`, `layer_idxs`, `qlw_flat`/offsets, etc.).
3. Constructs all `Region` objects from the flat data.
4. Links parent/child relationships by iterating `parent_ids`.
5. Identifies leaves as nodes with no children.

`Tree.perform_number_count(data, y)` routes data through the tree. It first runs a full forward pass to get precomputed activation patterns for all layers, then does a BFS over the tree nodes, routing points to children by matching their activation signature. This is efficient because the forward pass is vectorized.

`Tree.get_path_inequalities(region)` reconstructs the halfspace system `D * x ≤ g` on the fly from the stored weights and the region's activation path (traced via parent pointers). This mirrors `compute_path_geometry` in Julia.

---

## 8. Verification

`verification.jl` provides sanity checks (called via `run_verification.sh` or the Julia REPL):

- **Monte Carlo partition completeness**: samples random points and checks that exactly one region at a target layer contains each point.
- **Pairwise overlap scan**: uses AABB filtering + LP feasibility to detect overlapping regions.
- **Hierarchy check**: verifies that each child's Chebyshev center lies inside its parent's constraints.
- **Conditioning check**: computes Chebyshev radius of each region and flags thin/degenerate ones.

*Note*: `verification.jl` has some references to `region.Dlw_active` and `region.glw_active` that don't match the current `Region` struct (which has no such fields). These verification functions may not be fully working in the current codebase.

---

## 9. Entry Point: `run_geobin.jl`

The Julia entry point is `run_geobin.jl`. It:
1. Forces BLAS to single-threaded mode (to avoid nested parallelism with Julia threads).
2. Loads `Geobin` module everywhere (for distributed workers via `@everywhere`).
3. Opens the HDF5 file in read-write mode.
4. Scans for `epoch_*` groups, skips ones that already have `parent_ids` (unless `--overwrite`).
5. For each epoch: reads model weights, constructs the tree (sparse by default), optionally computes volumes, saves to HDF5, then runs GC.

CLI flags:
```
h5_file           path to the HDF5 file
--overwrite       recompute even if tree data exists
--exact_volume    use CDDLib exact volume (requires julia -p N)
--estimate_volume use volesti R estimation
--verbose         print tree summary tables
--exact_regions   use DFS facet-flipping instead of sparse construction
```

---

## 10. What I Am Uncertain About

1. **Output layer in tree**: It is unclear whether including the final (output) layer in the tree construction is intentional. If `hidden_sizes=[8,8]` and there are 2 classes, Julia sees 3 weight matrices and builds `L=3` layers of the tree. The deepest layer's "activation patterns" correspond to output logit signs, which effectively subdivide the input space by predicted class. This may be a meaningful design choice, or it may be a bug that doesn't matter because only intermediate layers are used downstream.

2. **`model/` subgroup**: `reconstruction.py` has conditional code for reading weights from a `model/` subgroup with `W_0`/`b_0` keys. The current Python training does not write this subgroup. I don't know if this code path is ever used or whether it's from an older version.

3. **`active_indices` contents**: For exact construction, `active_indices` stores the row indices in the *full* stacked `D * x ≤ g` (all layers concatenated) that are non-redundant. For sparse construction, it is empty and `get_path_inequalities` always returns the full constraint set. The Python code handles both cases (`if active_only and len(region.active_indices) > 0`).

4. **Verification functions**: Several functions in `verification.jl` reference struct fields (`Dlw_active`, `glw_active`) that don't exist on the current `Region` struct. These may have been part of an older design where constraints were stored directly on each node rather than recomputed on-the-fly.

5. **`centroids` shape**: The comment in Python says `"Centroids are (N, D) in the fixed Julia script"` which suggests there was a bug in older versions. I believe the current Julia code writes `centroids` as `(input_dim, num_nodes)` in column-major storage, which h5py reads as `(num_nodes, input_dim)`. But the exact shape convention after reading is worth verifying against a real HDF5 file.
