module Volume

using ..Regions
using ..Trees
using ..Utils: get_region_volume
using ProgressMeter
using RCall

export compute_volumes_parallel!, estimate_volumes_parallel!

"""
    compute_volumes_parallel!(tree::Tree; bound::Union{Float64, Nothing}=nothing)

Computes the EXACT volume of every region in the tree in parallel using CDDLib.
Updates the `volume` attribute of each region directly.
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
    p = Progress(n_regions; desc="Computing Exact Volumes (CDDLib): ")
    
    Threads.@threads for i in 1:n_regions
        vol = get_region_volume(all_regions[i]; bound=bound)
        all_regions[i].volume = vol
        next!(p)
    end
    
    return tree
end
"""
    estimate_volumes_parallel!(tree::Tree)

Estimates the volume of every region using the high-dimensional multiphase 
Monte Carlo samplers from the `volesti` R package.
Uses batching to provide a Julia progress bar while utilizing R's parallel `mclapply`.
"""
function estimate_volumes_parallel!(tree::Tree)
    # Load required R libraries
    R"""
    library(volesti)
    library(parallel)
    """

    # 1. Collect all regions
    all_regions = Region[]
    queue = [tree.root]
    
    while !isempty(queue)
        current_node = popfirst!(queue)
        push!(all_regions, current_node)
        append!(queue, current_node.children)
    end
    
    n_regions = length(all_regions)
    
    # Prepare lists to batch-send to R
    A_list = Matrix{Float64}[]
    b_list = Vector{Float64}[]
    valid_indices = Int[]
    
    for i in 1:n_regions
        region = all_regions[i]
        A, b = get_path_inequalities(region)
        
        if size(A, 1) == 0
            region.volume = Inf # Root/unconstrained region
        else
            push!(A_list, Float64.(A))
            push!(b_list, Float64.(b))
            push!(valid_indices, i)
        end
    end
    
    n_valid = length(valid_indices)
    if n_valid == 0
        return tree
    end

    n_threads = Threads.nthreads()
    @rput n_threads
    
    # 2. Setup Julia Progress Bar & Chunking
    chunk_size = 50 # Process 50 regions per R call
    chunks = collect(Iterators.partition(1:n_valid, chunk_size))

    println("   - Prepared $n_valid regions for volume estimation in $(length(chunks)) chunks using $n_threads threads.")
    
    p = Progress(n_valid; desc="Estimating Volumes (VolEsti): ")
    est_vols_all = Float64[]

    # 3. Process chunks sequentially in Julia, but in parallel inside R
    for chunk in chunks
        # Extract the current batch
        A_chunk = A_list[chunk]
        b_chunk = b_list[chunk]
        
        # Send only this batch to R
        @rput A_chunk
        @rput b_chunk
        
        R"""
        # R computes just this chunk in parallel
        vols <- mclapply(1:length(A_chunk), function(i) {
            poly <- tryCatch(
                Hpolytope(A = A_chunk[[i]], b = b_chunk[[i]]), 
                error = function(e) NULL
            )
            if (is.null(poly)) return(0.0)
            
            vol <- tryCatch(
                volume(poly), 
                error = function(e) 0.0
            )
            return(vol)
        }, mc.cores = n_threads)
        
        est_vols_chunk <- unlist(vols)
        """
        
        # Pull chunk results back to Julia
        @rget est_vols_chunk
        append!(est_vols_all, est_vols_chunk)
        
        # Step the progress bar by the chunk size
        ProgressMeter.next!(p; step=length(chunk))
    end
    
    # 4. Assign the results back to the tree regions
    for (list_idx, region_idx) in enumerate(valid_indices)
        all_regions[region_idx].volume = est_vols_all[list_idx]
    end
    
    return tree
end

end # module