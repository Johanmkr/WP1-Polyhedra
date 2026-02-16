module Construction

using ..Regions
using ..Trees
using ..Geometry
using LinearAlgebra
using Statistics
using ProgressMeter
using JuMP
using HiGHS

export construct_tree!

# Data structure for candidates
struct RegionCandidate
    activation_pattern::Vector{Int}
    D::Matrix{Float64}
    g::Vector{Float64}
    A_next::Matrix{Float64}
    c_next::Vector{Float64}
    active_indices::Vector{Int}
    is_bounded::Bool
    volume::Float64
end

function construct_tree!(tree::Tree; verbose::Bool=false)
    current_layer_regions = [tree.root]
    
    # --- OPTIMIZATION: THREAD-LOCAL MODELS ---
    # FIX: Use maxthreadid() instead of nthreads(). 
    # Thread IDs can be > nthreads() (e.g., if using interactive threads or sparse pools).
    max_tid = Threads.maxthreadid()
    models = [Model(HiGHS.Optimizer) for _ in 1:max_tid]
    
    for m in models
        set_silent(m)
    end
    # -----------------------------------------

    for layer_idx in 1:tree.L
        W = tree.weights[layer_idx]
        b = tree.biases[layer_idx]

        if verbose
            println("Layer $layer_idx: Processing $(length(current_layer_regions)) regions...")
        end

        # Pre-allocate results array to avoid race conditions
        results = Vector{Vector{Region}}(undef, length(current_layer_regions))

        Threads.@threads for i in eachindex(current_layer_regions)
            # Use the model dedicated to this thread ID
            # This is now safe even if threadid() returns 17
            local_model = models[Threads.threadid()]
            
            parent = current_layer_regions[i]
            results[i] = process_parent_region(parent, W, b, layer_idx, local_model)
        end
        
        next_layer_regions = reduce(vcat, results)
        current_layer_regions = next_layer_regions
        
        if isempty(current_layer_regions)
            verbose && println("Tree construction stopped early at layer $layer_idx.")
            break
        end
    end
    
    return tree
end

function process_parent_region(parent::Region, W::Matrix, b::Vector, layer_idx::Int, model::Model)
    # 1. Fetch FULL ancestry
    D_ancestry, g_ancestry = get_path_inequalities(parent)

    # 2. Find Candidates (Passing model for reuse)
    candidates = find_region_candidates(
        D_ancestry, g_ancestry,
        parent.Alw, parent.clw, parent.x,
        W, b,
        model
    )

    children = Region[]
    for cand in candidates
        # 3. Check Feasibility
        feasible_point = get_feasible_point(
            [D_ancestry; cand.D], 
            [g_ancestry; cand.g];
            model = model
        )

        if !isnothing(feasible_point)
            child = Region(cand.activation_pattern)
            
            child.q_tilde     = cand.active_indices
            child.Dlw         = cand.D
            child.glw         = cand.g
            
            if isempty(cand.active_indices)
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
            child.bounded       = cand.is_bounded
            child.volume        = cand.volume

            add_child!(parent, child)
            push!(children, child)
        end
    end
    return children
end

function find_region_candidates(D_prev, g_prev, A_prev, c_prev, x_start, W, b, model)
    z_start = W * (A_prev * x_start + c_prev) + b
    q_start = Int.(z_start .> 0)

    queue = [q_start]
    visited = Set{Vector{Int}}([q_start])
    candidates = RegionCandidate[]

    while !isempty(queue)
        q_curr = popfirst!(queue)
        D, g, A_next, c_next = calculate_layer_geometry(W, b, q_curr, A_prev, c_prev)

        # Use LP-based Active Set check (Faster & Robust)
        active_indices, is_bounded, vol = Geometry.find_active_indices_exact(D, g, D_prev, g_prev)

        push!(candidates, RegionCandidate(q_curr, D, g, A_next, c_next, active_indices, is_bounded, vol))

        for idx in active_indices
            q_neighbor = copy(q_curr)
            q_neighbor[idx] = 1 - q_neighbor[idx]
            if !(q_neighbor in visited)
                push!(visited, q_neighbor)
                push!(queue, q_neighbor)
            end
        end
    end
    return candidates
end

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

end # module