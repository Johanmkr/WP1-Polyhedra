module Construction

using ..Regions
using ..Trees
using ..Geometry
using LinearAlgebra
using Statistics
using ProgressMeter

export construct_tree!

# 1. Lightweight struct for intermediate calculation results
# This avoids the overhead and messiness of Dict{String, Any}
struct RegionCandidate
    activation_pattern::Vector{Int}
    # Local inequality constraints (D x <= g)
    D::Matrix{Float64}
    g::Vector{Float64}
    # Affine map for the next layer (A_next x + c_next)
    A_next::Matrix{Float64}
    c_next::Vector{Float64}
    # Indices of active constraints from the previous layer
    active_indices::Vector{Int}
end

function construct_tree!(tree::Tree; verbose::Bool=false)
    current_layer_regions = [tree.root]

    for layer_idx in 1:tree.L
        W = tree.weights[layer_idx]
        b = tree.biases[layer_idx]

        if verbose
            println("Layer $layer_idx: Processing $(length(current_layer_regions)) regions...")
        end

        # Parallel map: Process each parent to find its children
        # We use Threads.@spawn or simply @threads with a reduction. 
        # Here is a robust map-reduce pattern that avoids locks.
        next_layer_regions = Regions.Region[] # Pre-allocation hint if possible
        
        # Flatten the results of the parallel loop safely
        results = Vector{Vector{Region}}(undef, length(current_layer_regions))

        Threads.@threads for i in eachindex(current_layer_regions)
        # for i in eachindex(current_layer_regions)
            parent = current_layer_regions[i]
            results[i] = process_parent_region(parent, W, b, layer_idx)
        end
        
        # Combine all children found in this layer
        next_layer_regions = reduce(vcat, results)

        # Update tree state
        current_layer_regions = next_layer_regions
        
        if isempty(current_layer_regions)
            verbose && println("Tree construction stopped early at layer $layer_idx (no valid regions).")
            break
        end
    end
    
    return tree
end

function process_parent_region(parent::Region, W::Matrix, b::Vector, layer_idx::Int)
    # FIX 1: Retrieve FULL ancestry constraints (Root + Grandparents + Parent)
    D_ancestry, g_ancestry = get_path_inequalities(parent)

    # FIX 2: Pass FULL ancestry to candidate finder
    # This ensures 'active_indices' are calculated relative to the Global Space, 
    # not just the immediate parent.
    candidates = find_region_candidates(
        D_ancestry, g_ancestry,       # <--- CHANGED from parent.Dlw_active, parent.glw_active
        parent.Alw, parent.clw, parent.x,
        W, b
    )

    children = Region[]
    for cand in candidates
        # FIX 3: Check feasibility against FULL ancestry + New Cuts
        feasible_point = get_feasible_point([D_ancestry; cand.D], [g_ancestry; cand.g])

        if !isnothing(feasible_point)
            child = Region(cand.activation_pattern)
            
            child.q_tilde     = cand.active_indices
            child.Dlw         = cand.D
            child.glw         = cand.g
            
            # Logic for active constraints
            if isempty(cand.active_indices)
                # If filter says "None", we can either store NONE or ALL.
                # Storing ALL (cand.D) is safer in case of numerical filter failure.
                # Storing NONE is more efficient if we trust the filter.
                # Let's stick to the original "Safe" approach:
                child.Dlw_active = cand.D 
                child.glw_active = cand.g
            else
                child.Dlw_active  = cand.D[cand.active_indices, :]
                child.glw_active  = cand.g[cand.active_indices]
            end
            
            child.Alw          = cand.A_next
            child.clw          = cand.c_next
            child.layer_number = layer_idx
            child.x            = feasible_point

            add_child!(parent, child)
            push!(children, child)
        end
    end
    return children
end

# 3. BFS / Enumeration Logic
function find_region_candidates(D_prev, g_prev, A_prev, c_prev, x_start, W, b)
    
    # Initial Activation at the starting point
    z_start = W * (A_prev * x_start + c_prev) + b
    q_start = Int.(z_start .> 0)

    # Queue for BFS
    queue = [q_start]
    visited = Set{Vector{Int}}([q_start])
    
    candidates = RegionCandidate[]

    while !isempty(queue)
        q_curr = popfirst!(queue)

        # Calculate geometric quantities for this activation pattern
        D, g, A_next, c_next = calculate_layer_geometry(W, b, q_curr, A_prev, c_prev)

        # Identify active constraints (Geometric filtering)
        # This function returns indices of D that are actually touching the region boundaries
        active_indices, _ = find_active_indices_exact(D, g, D_prev, g_prev)

        # Store valid candidate
        push!(candidates, RegionCandidate(q_curr, D, g, A_next, c_next, active_indices))

        # Explore neighbors
        # We only flip bits corresponding to ACTIVE constraints. 
        # If a constraint is not active, crossing it implies leaving the feasible set entirely or redundancy.
        for idx in active_indices
            q_neighbor = copy(q_curr)
            q_neighbor[idx] = 1 - q_neighbor[idx] # Flip bit (0->1 or 1->0)

            if !(q_neighbor in visited)
                push!(visited, q_neighbor)
                push!(queue, q_neighbor)
            end
        end
    end

    return candidates
end

# 4. Extracted Math Helper (keeps the main loop clean)
function calculate_layer_geometry(W, b, q, A_prev, c_prev)
    W_hat = W * A_prev
    b_hat = W * c_prev + b
    s_vec = -2.0 .* q .+ 1.0
    D = s_vec .* W_hat
    g = -(s_vec .* b_hat)
    A = q .* W_hat
    c = q .* b_hat 

    return D, g, A, c
end


# function calculate_next_layer_quantities(Wl, bl, qlw, Alw_prev, clw_prev)
#     Wl_hat = Wl * Alw_prev
#     bl_hat = Wl * clw_prev + bl
#     s_vec = -2.0 .* qlw .+ 1.0
#     Dlw = s_vec .* Wl_hat
#     glw = -(s_vec .* bl_hat)
#     Alw = qlw .* Wl_hat
#     clw = qlw .* bl_hat
#     return Dlw, glw, Alw, clw
# end

end # module