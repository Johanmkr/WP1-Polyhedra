module Volume

using ..Regions
using ..Trees
using ..Utils: get_region_volume
using ProgressMeter
using RCall
using JuMP
using HiGHS
using Distributed
using ProgressMeter

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
        all_regions[i].volume = volumes[i]
    end
    
    return tree
end

"""
Checks if a region is unbounded by evaluating its recession cone.
Re-uses an existing JuMP model to eliminate allocation overhead.
"""
function is_region_unbounded_fast(A::Matrix{Float64}, model::Model)
    dim = size(A, 2)
    # If there are no constraints, the whole space is unbounded
    if size(A, 1) == 0
        return true
    end
    
    empty!(model) # Clears variables and constraints from previous runs
    
    # Bounded test vector d
    @variable(model, -1.0 <= d[1:dim] <= 1.0)
    # Recession cone constraint
    @constraint(model, A * d .<= 0)
    
    for i in 1:dim
        # Check if we can move infinitely in the positive direction of axis i
        @objective(model, Max, d[i])
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL && objective_value(model) > 1e-6
            return true
        end
        
        # Check if we can move infinitely in the negative direction of axis i
        @objective(model, Min, d[i])
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL && objective_value(model) < -1e-6
            return true
        end
    end
    return false
end

"""
    estimate_volumes_parallel!(tree::Tree; chunk_size::Int=200)

Estimates the volume of every region using `volesti` in R.
Optimized using a thread-safe JuMP model pool for recession cone checks, 
minimizing allocation overhead while keeping memory usage strictly bounded.
"""
function estimate_volumes_parallel!(tree::Tree; chunk_size::Int=200)
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
    # Creates one model per thread to avoid allocation bottlenecks
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

    # 2. Iterate in chunks
    for chunk_start in 1:chunk_size:n_regions
        chunk_end = min(chunk_start + chunk_size - 1, n_regions)
        current_batch = all_regions[chunk_start:chunk_end]
        batch_size = length(current_batch)
        
        A_list = Vector{Matrix{Float64}}(undef, batch_size)
        b_list = Vector{Vector{Float64}}(undef, batch_size)
        is_bounded_flags = zeros(Bool, batch_size)
        
        # --- PHASE A: Multi-threaded Julia LP Checks ---
        Threads.@threads for local_idx in 1:batch_size
            region = current_batch[local_idx]
            A, b = get_path_inequalities(region)
            A_list[local_idx] = A
            b_list[local_idx] = b
            
            if size(A, 1) == 0
                region.volume = Inf
                region.bounded = false
            else
                local_model = take!(model_pool)
                try
                    unbounded = is_region_unbounded_fast(A, local_model)
                    if unbounded
                        region.volume = Inf
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
                current_batch[local_idx].volume = est_vols_chunk[k]
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