module Pruning

using ..Regions
using ..Trees
using ..Geometry
using ..Utils
using LinearAlgebra
using Polyhedra
using CDDLib
using ProgressMeter

export prune_tree!

"""
    prune_tree!(tree::Tree; method=:volume, bound=nothing, tol=1e-8)

Recursively removes regions from the tree that are "empty" or have negligible volume.
- :volume method calculates exact volume (safer for 'thin' but valid regions).
- :feasibility method uses LP to find an interior point (faster).
"""
function prune_tree!(tree::Tree; method=:volume, bound=nothing, tol=1e-8)
    println("\n✂️  Pruning Tree (Method: $method)...")
    
    # Statistics to track what we removed
    stats = Dict(
        "nodes_visited" => 0,
        "nodes_kept" => 0,
        "nodes_pruned" => 0
    )
    
    # Start recursive pruning from the root
    # We assume the root itself is valid (usually the whole space)
    _prune_recursive!(tree.root, method, bound, tol, stats)
    
    # Update Tree Metadata (optional, depending on your Tree struct)
    # tree.total_nodes = stats["nodes_kept"] 
    
    println("\n✅ Pruning Complete:")
    println("   - Visited: $(stats["nodes_visited"])")
    println("   - Kept:    $(stats["nodes_kept"])")
    println("   - Pruned:  $(stats["nodes_pruned"]) branches")
    
    return tree
end

function _prune_recursive!(parent::Region, method, bound, tol, stats)
    # 1. Identify valid children
    valid_children = Region[]
    
    # If parent has no children, we are at a leaf (or dead end)
    if isempty(parent.children)
        return
    end

    for child in parent.children
        stats["nodes_visited"] += 1
        
        is_full = false
        
        if method == :volume
            # Uses your existing volume logic
            vol = get_region_volume(child, bound=bound)
            is_full = (vol == Inf || vol > tol)
        elseif method == :feasibility
            # Fast LP check: Can we fit a ball of radius 'tol' inside?
            # We assume you have the robust 'get_chebyshev_center' from previous discussions
            center = get_chebyshev_center(child.Dlw, child.glw)
            # If center exists and radius > tol (implicit in get_chebyshev_center logic), it's full
            is_full = !isnothing(center)
        end

        if is_full
            push!(valid_children, child)
            stats["nodes_kept"] += 1
            
            # 2. Recurse ONLY on valid children
            # (If a child is empty, its subtree is automatically empty, so we don't recurse)
            _prune_recursive!(child, method, bound, tol, stats)
        else
            stats["nodes_pruned"] += 1
        end
    end
    
    # 3. Update the parent's children list to only contain valid regions
    parent.children = valid_children
end

# --- Helper for Feasibility Mode (if not already defined) ---
# This is much faster than volume computation for pruning
using JuMP, HiGHS

function get_chebyshev_center(A::Matrix{Float64}, b::Vector{Float64}; limit=1e5, tol=1e-7)
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -limit <= x[1:n] <= limit)
    @variable(model, r) # Radius

    # Geometric constraint: a_i*x + ||a_i||*r <= b_i
    for i in 1:m
        a_norm = norm(A[i, :])
        if a_norm > 1e-8
            @constraint(model, dot(A[i, :], x) + r * a_norm <= b[i])
        elseif b[i] < -1e-8
            return nothing # Infeasible constraint 0 <= negative
        end
    end

    @objective(model, Max, r)
    optimize!(model)

    if termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        # If max radius is positive, a full-dimensional region exists
        if value(r) > tol
            return value.(x)
        end
    end
    return nothing
end

end # module