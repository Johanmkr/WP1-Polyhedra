module SparseConstruction

using ..Regions
using ..Trees
using ..Construction: calculate_layer_geometry
using LinearAlgebra
using ProgressMeter
using Base.Threads: ReentrantLock

export construct_tree_sparse!, construct_tree_sparse_mc!

"""
    compute_activation_path(tree::Tree, x0::Vector{Float64})

Performs a forward pass for a single point, returning the sequence of 
activation signatures (q) for every layer.
"""
function compute_activation_path(tree::Tree, x0::AbstractVector{Float32})
    q_path = Vector{Vector{Int}}(undef, tree.L)
    a = x0
    
    for l in 1:tree.L
        W = tree.weights[l]
        b = tree.biases[l]
        
        # Pre-activation
        z = W * a + b
        
        # Activation signature
        q = Int.(z .> 0)
        q_path[l] = q
        
        # Post-activation (ReLU)
        a = q .* z 
    end
    
    return q_path
end


"""
    insert_path!(tree::Tree, x0::Vector{Float64}, q_path::Vector{Vector{Int}}, root_lock::ReentrantLock)
"""
function insert_path!(tree::Tree, x0::AbstractVector{Float32}, q_path::Vector{Vector{Int}}, root_lock::ReentrantLock)
    current_node = tree.root
    
    # Track A and c dynamically to save memory
    A_curr = tree.root.Alw
    c_curr = tree.root.clw
    
    for l in 1:tree.L
        q = q_path[l]
        
        # 1. Check if this region exists in the current node's children
        existing_child = nothing
        for child in get_children(current_node)
            if child.qlw == q
                existing_child = child
                break
            end
        end
        
        W = tree.weights[l]
        b = tree.biases[l]
        
        if !isnothing(existing_child)
            # Path exists, calculate A and c to propagate down without storing
            if l != tree.L
                _, _, A_curr, c_curr = calculate_layer_geometry(W, b, q, A_curr, c_curr)
            end
            current_node = existing_child
        else
            # 2. Path doesn't exist, create a new Region
            new_node = Region(q)
            new_node.layer_number = l
            new_node.x = x0 
            
            # 3. Compute exact geometry (Done in parallel!)
            D, g, A_next, c_next = calculate_layer_geometry(W, b, q, A_curr, c_curr)
            
            new_node.Dlw = D
            new_node.glw = g
            new_node.Dlw_active = D
            new_node.glw_active = g
            
            # MEMORY SAVER: Don't store Alw and clw for leaf nodes.
            if l == tree.L
                new_node.Alw = Matrix{Float64}(undef, 0, 0)
                new_node.clw = Float64[]
            else
                new_node.Alw = A_next
                new_node.clw = c_next
            end
            
            # 4. Safely attach the node
            if l == 1
                # Only lock when pushing to the shared root node
                lock(root_lock) do
                    add_child!(current_node, new_node)
                end
            else
                # For l > 1, the branch batching guarantees absolute thread isolation
                # No other thread will ever touch this current_node, so no lock needed!
                add_child!(current_node, new_node)
            end
            
            current_node = new_node
            
            # Update for next iteration
            A_curr = A_next
            c_curr = c_next
        end
    end
end

function construct_tree_sparse!(tree::Tree, points::Matrix{Float32})
    dim, n_points = size(points)
    @assert dim == tree.input_dim "Point dimensions do not match tree input dimension."
    println("\n⚡ Starting Sparse Tree Construction...")
    println("   - Points to process: $n_points")
    println("   - Threads available: $(Threads.nthreads())")
    
    # --- PHASE 1: Parallel Forward Pass ---
    paths = Vector{Vector{Vector{Int}}}(undef, n_points)
    
    p1 = Progress(n_points; desc="Computing Paths (Parallel): ")
    Threads.@threads for i in 1:n_points
        paths[i] = compute_activation_path(tree, @view(points[:, i]))
        next!(p1)
    end
    
    # --- PHASE 2: Group by Layer 1 Branch ---
    branch_groups = Dict{Vector{Int}, Vector{Int}}()
    for i in 1:n_points
        q1 = paths[i][1]
        if !haskey(branch_groups, q1)
            branch_groups[q1] = Int[]
        end
        push!(branch_groups[q1], i)
    end
    
    unique_branches = collect(branch_groups)
    println("   - Partitioned into $(length(unique_branches)) independent branches.")
    
    # --- PHASE 3: Batched Parallel Assembly with Root Lock ---
    n_threads = Threads.nthreads()
    root_lock = ReentrantLock() # Create the lock for the root node
    
    # Create empty batches for each thread
    batches = [Vector{Pair{Vector{Int}, Vector{Int}}}() for _ in 1:n_threads]
    
    # Distribute branches evenly across threads (Round Robin)
    for (i, branch) in enumerate(unique_branches)
        batch_idx = mod1(i, n_threads)
        push!(batches[batch_idx], branch)
    end
    
    p2 = Progress(n_points; desc="Assembling Subtrees (Batched Parallel): ")
    
    Threads.@threads for batch in batches
        for (q1, indices) in batch
            for idx in indices
                # Pass the root lock to the insertion function
                insert_path!(tree, @view(points[:, idx]), paths[idx], root_lock)
                next!(p2)
            end
        end
        # Keep RAM usage flat
        GC.gc(false)
    end
    
    println("✅ Sparse tree construction complete.")
    return tree
end

"""
    construct_tree_sparse_mc!(tree::Tree, num_points::Int, bound::Float64)

Wrapper for Monte Carlo sampling. Generates uniform random points within 
a specified [-bound, bound] hypercube and builds the tree.
"""
function construct_tree_sparse_mc!(tree::Tree, num_points::Int, bound::Float64)
    # Generate random points in the hypercube
    points = (rand(tree.input_dim, num_points) .- 0.5) .* (2 * bound)
    return construct_tree_sparse!(tree, points)
end

end # module