module Geobin

using LinearAlgebra
using JuMP
using HiGHS
using ProgressMeter
using Polyhedra
using CDDLib
using Statistics
using Printf

# Include submodules
include("utils.jl")
include("region.jl")
include("geometry.jl")
include("tree.jl")
include("construction.jl")
include("verification.jl")

# Use and re-export submodules
using .Utils
using .Regions
using .Geometry
using .Trees
using .Construction
using .Verification

export Region, Tree, construct_tree!, get_regions_at_layer, get_path_inequalities
export verify_volume_conservation, check_point_partition, get_region_volume, scan_all_overlaps_strict
export find_hyperplanes

end # module
