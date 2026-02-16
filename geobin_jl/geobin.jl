module Geobin

using LinearAlgebra
using JuMP
using HiGHS
using ProgressMeter
using Polyhedra
using CDDLib
using Statistics
using Printf

# Include order
include("region.jl")    
include("geometry.jl")  
include("utils.jl")     
include("tree.jl")      
include("construction.jl")
include("verification.jl")
include("pruning.jl")
include("save_tree.jl")

# Use and re-export
using .Regions
using .Utils
using .Geometry
using .Trees
using .Construction
using .Verification
using .Pruning
using .SaveTree

# EXPORT EVERYTHING
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

# CRITICAL FIX: Export the new save function
export save_single_tree_to_hdf5

end # module