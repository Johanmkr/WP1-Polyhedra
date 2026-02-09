module PolyhedraTree

using LinearAlgebra
using JuMP
using HiGHS
using ProgressMeter
using Polyhedra
using CDDLib
using Statistics
using Printf

import Base: show

export Region, Tree, construct_tree!, get_regions_at_layer, get_path_inequalities
export verify_volume_conservation, check_point_partition, get_region_volume, scan_all_overlaps_strict

include("utils.jl")
include("geometry.jl")
include("region.jl")
include("tree.jl")
include("construction.jl")
include("verification.jl")

end # module