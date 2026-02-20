module Geobin

using LinearAlgebra
using JuMP
using HiGHS
using ProgressMeter
using Polyhedra
using CDDLib
using Statistics
using Printf

# --- FIX: Load Order Changed ---
# Regions must be loaded first because Utils and Trees depend on it.
include("region.jl")    # <--- Moved up
include("geometry.jl")  # Independent
include("utils.jl")     # Depends on Regions
include("tree.jl")      # Depends on Regions, Utils
include("construction.jl")
include("verification.jl")
include("pruning.jl")
include("save_tree.jl")
include("sparse_construction.jl")

# Use and re-export submodules
using .Regions
using .Utils
using .Geometry
using .Trees
using .Construction
using .Verification
using .Pruning
using .SaveTree
using .SparseConstruction

export Region, Tree, construct_tree!, get_regions_at_layer, print_tree_summary, get_path_inequalities
export verify_volume_conservation, check_point_partition, get_region_volume, scan_all_overlaps_strict
export find_hyperplanes
export prune_tree!, analyze_region
export verify_tree_properties
export verify_partition_completeness_monte_carlo
export scan_all_overlaps_strict_optimized
export verify_tree_hierarchy_robust
export check_region_conditioning
export find_active_indices_lp
export save_tree_to_h5, read_state_dict_from_h5
export construct_tree_sparse!, construct_tree_sparse_mc!

end # module