module Volume

using ..Regions
using ..Trees
using ..Utils   
using ProgressMeter
using RCall
using JuMP
using HiGHS
using Distributed
using LinearAlgebra

export compute_volumes_parallel!, estimate_volumes_parallel!

"""
    compute_volumes_parallel!(tree::Tree; bound::Union{Float64, Nothing}=nothing)

Computes the EXACT volume of every region in the tree in parallel using CDDLib.
Uses `Distributed.pmap` to safely isolate the non-thread-safe CDDLib C-library.
"""
function compute_volumes_parallel!(tree::Tree; bound::Union{Float64, Nothing}=nothing)
    all_regions = Region[]
    queue = [tree.root]
    
    while !isempty(queue)
        current_node = popfirst!(queue)
        push!(all_regions, current_node)
        append!(queue, current_node.children)
    end
    
    n_regions = length(all_regions)
    
    # Check if worker processes are available
    if nworkers() <= 1
        @warn "Only 1 worker process available. Run Julia with `julia -p auto` for parallelism."
    end
    
    println("Computing Exact Volumes (CDDLib) using $(nworkers()) worker processes...")

    # Use pmap to safely distribute the CDDLib workload across isolated processes
    volumes = @showprogress pmap(1:n_regions) do i
        # 1. Turn OFF Garbage Collection so it doesn't interrupt CDDLib
        GC.enable(false)
        
        try
            vol = get_region_volume(all_regions[i]; bound=bound)
            return vol
        catch e
            return 0.0 # Return 0 if triangulation completely fails mathematically
        finally
            # 2. Turn GC back ON and force a safe cleanup between regions
            GC.enable(true)
            GC.gc()
        end
    end
    
    # Assign the calculated volumes back to the main tree
    for i in 1:n_regions
        all_regions[i].volume_es = volumes[i]
    end
    
    return tree
end

"""
Checks if a region is unbounded by evaluating its recession cone {d | Ad <= 0}.
Uses strict row-normalization, higher tolerances, and post-solve validation to prevent 
numerical noise from falsely classifying bounded regions as unbounded.
"""
function is_region_unbounded_fast(A::Matrix{Float64}, model::Model; tol=1e-4)
    dim = size(A, 2)
    if size(A, 1) == 0
        return true
    end
    
    # 1. Normalize A to prevent massive scaling differences from confusing the LP solver
    A_norm = copy(A)
    for i in 1:size(A_norm, 1)
        row_norm = norm(A_norm[i, :])
        if row_norm > 1e-12
            A_norm[i, :] ./= row_norm
        end
    end

    empty!(model)
    
    # Bounded test vector
    @variable(model, -1.0 <= d[1:dim] <= 1.0)
    
    # We allow a tiny numerical violation (1e-8) during the solve, 
    # but we will strictly verify it later.
    @constraint(model, A_norm * d .<= 1e-8)
    
    for i in 1:dim
        # --- Check positive direction ---
        @objective(model, Max, d[i])
        optimize!(model)
        
        status = termination_status(model)
        if status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
            if objective_value(model) > tol
                # STRICT DOUBLE CHECK: Extract the actual vector and mathematically prove it
                d_vec = value.(d)
                if maximum(A_norm * d_vec) <= tol && norm(d_vec) > tol
                    return true # It is definitively unbounded
                end
            end
        end
        
        # --- Check negative direction ---
        @objective(model, Min, d[i])
        optimize!(model)
        
        status = termination_status(model)
        if status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
            if objective_value(model) < -tol
                # STRICT DOUBLE CHECK: Extract the actual vector and mathematically prove it
                d_vec = value.(d)
                if maximum(A_norm * d_vec) <= tol && norm(d_vec) > tol
                    return true # It is definitively unbounded
                end
            end
        end
    end
    
    return false
end

"""
    estimate_volumes_parallel!(tree::Tree; chunk_size::Int=200, hypercube_bounds::Tuple{Float64, Float64}=(0.0, 1.0))

Estimates the volume of every region using `volesti` in R.
Bounds the space to a hypercube (default [0, 1]^d) before estimation.
Optimized using a thread-safe JuMP model pool for recession cone checks, 
minimizing allocation overhead while keeping memory usage strictly bounded.
"""
function estimate_volumes_parallel!(tree::Tree; chunk_size::Int=200, hypercube_bounds::Tuple{Float64, Float64}=(0.0, 1.0))
    # 1. Flatten tree
    all_regions = Region[]
    queue = [tree.root]
    
    while !isempty(queue)
        current_node = popfirst!(queue)
        push!(all_regions, current_node)
        append!(queue, current_node.children)
    end
    
    n_regions = length(all_regions)
    n_threads = Threads.nthreads()
    
    # --- SETUP JUMP MODEL POOL ---
    model_pool = Channel{Model}(n_threads)
    for _ in 1:n_threads
        m = Model(HiGHS.Optimizer)
        set_silent(m)
        put!(model_pool, m)
    end
    
    println("Estimating volumes for $n_regions regions using $n_threads threads in Julia & R...")
    
    @rput n_threads

    R"""
    library(volesti)
    library(parallel)
    cl <- makeCluster(n_threads, type="PSOCK")
    clusterEvalQ(cl, library(volesti))
    """
    
    p = Progress(n_regions; desc="Estimating Volumes: ")
    lower_bnd, upper_bnd = hypercube_bounds

    # 2. Iterate in chunks
    for chunk_start in 1:chunk_size:n_regions
        chunk_end = min(chunk_start + chunk_size - 1, n_regions)
        current_batch = all_regions[chunk_start:chunk_end]
        batch_size = length(current_batch)
        
        A_list = Vector{Matrix{Float64}}(undef, batch_size)
        b_list = Vector{Vector{Float64}}(undef, batch_size)
        is_bounded_flags = zeros(Bool, batch_size)
        
        # --- PHASE A: Multi-threaded Julia LP Checks & Bounding ---
        Threads.@threads for local_idx in 1:batch_size
            region = current_batch[local_idx]
            q_path = get_activation_path(region)
            A, b = compute_path_geometry(tree.weights, tree.biases, q_path; active_indices=region.active_indices)
            
            dim = size(A, 2)
            
            # Apply Hypercube Bounds
            if dim > 0
                I_mat = Matrix{Float64}(I, dim, dim)
                # Append constraints: I*x <= upper_bnd and -I*x <= -lower_bnd
                A = vcat(A, I_mat, -I_mat)
                b = vcat(b, fill(upper_bnd, dim), fill(-lower_bnd, dim))
            end

            A_list[local_idx] = A
            b_list[local_idx] = b
            
            if size(A, 1) == 0
                region.volume_es = Inf
                region.bounded = false
            else
                local_model = take!(model_pool)
                try
                    # Note: Because we added the hypercube bounds, this will effectively 
                    # always return false (bounded), but keeping it acts as a safe geometric check
                    # in case the resulting intersection is mathematically degenerate.
                    unbounded = is_region_unbounded_fast(A, local_model)
                    if unbounded
                        region.volume_es = Inf
                        region.bounded = false
                    else
                        region.bounded = true
                        is_bounded_flags[local_idx] = true
                    end
                finally
                    put!(model_pool, local_model)
                end
            end
        end
        
        # --- PHASE B: Collect Valid Matrices for R ---
        A_chunk = Matrix{Float64}[]
        b_chunk = Vector{Float64}[]
        valid_local_indices = Int[]
        
        for local_idx in 1:batch_size
            if is_bounded_flags[local_idx]
                push!(A_chunk, Float64.(A_list[local_idx]))
                push!(b_chunk, Float64.(b_list[local_idx]))
                push!(valid_local_indices, local_idx)
            end
        end
        
        # --- PHASE C: Multi-threaded R Volesti Estimation ---
        if !isempty(valid_local_indices)
            @rput A_chunk
            @rput b_chunk
            
            R"""
            vols <- parLapply(cl, 1:length(A_chunk), function(i, A_list, b_list) {
                poly <- tryCatch(
                    Hpolytope(A = A_list[[i]], b = b_list[[i]]), 
                    error = function(e) NULL
                )
                if (is.null(poly)) return(0.0)
                
                return(tryCatch(volume(poly), error = function(e) 0.0))
            }, A_list = A_chunk, b_list = b_chunk)
            
            est_vols_chunk <- unlist(vols)
            rm(A_chunk, b_chunk, vols)
            gc()
            """
            
            @rget est_vols_chunk
            
            for (k, local_idx) in enumerate(valid_local_indices)
                current_batch[local_idx].volume_es = est_vols_chunk[k]
            end
        end
        
        ProgressMeter.next!(p; step=batch_size)
    end

    R"""
    stopCluster(cl)
    """
    
    return tree
end

end # module