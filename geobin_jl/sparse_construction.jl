module SparseConstruction

using ..Regions
using ..Trees
using ..Construction: calculate_layer_geometry
using LinearAlgebra
using ProgressMeter

export construct_tree_sparse!, construct_tree_sparse_mc!

"""
    compute_activation_path(tree::Tree, x0::Vector{Float64})

Performs a forward pass for a single point, returning the sequence of 
activation signatures (q) for every layer.
"""
function compute_activation_path(tree::Tree, x0::Vector{Float64})
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
    insert_path!(tree::Tree, x0::Vector{Float64}, q_path::Vector{Vector{Int}})

Takes a pre-computed activation path and adds it to the tree. 
If a region already exists, it traverses it. If not, it creates a new region 
and computes the necessary affine geometries to maintain compatibility with 
existing downstream verification/volume tools.
"""
function insert_path!(tree::Tree, x0::Vector{Float64}, q_path::Vector{Vector{Int}})
    current_node = tree.root
    
    for l in 1:tree.L
        q = q_path[l]
        
        # 1. Check if this region (activation pattern) already exists
        existing_child = nothing
        for child in get_children(current_node)
            if child.qlw == q
                existing_child = child
                break
            end
        end
        
        if !isnothing(existing_child)
            # Path exists, just traverse down
            current_node = existing_child
        else
            # 2. Path doesn't exist, create a new Region
            new_node = Region(q)
            new_node.layer_number = l
            new_node.x = x0 # The input point is a guaranteed feasible point for this region
            
            # 3. Compute exact geometry to retain compatibility with geobin tools
            W = tree.weights[l]
            b = tree.biases[l]
            
            # Re-use your existing geometry calculation
            D, g, A_next, c_next = calculate_layer_geometry(
                W, b, q, current_node.Alw, current_node.clw
            )
            
            new_node.Dlw = D
            new_node.glw = g
            new_node.Alw = A_next
            new_node.clw = c_next
            
            # Note: The exact method uses LPs to filter Dlw into Dlw_active. 
            # To keep sparse construction fast, we skip the LP redundancy check and 
            # treat all local inequalities as active. This is mathematically safe for 
            # volume calculation and overlap checking.
            new_node.Dlw_active = D
            new_node.glw_active = g
            
            # Link node into tree
            add_child!(current_node, new_node)
            current_node = new_node
        end
    end
end

"""
    construct_tree_sparse!(tree::Tree, points::Matrix{Float64})

Takes a matrix of points (size: input_dim x num_points) and builds the tree
using a fast thread-safe parallel approach.
"""
function construct_tree_sparse!(tree::Tree, points::Matrix{Float64})
    dim, n_points = size(points)
    @assert dim == tree.input_dim "Point dimensions do not match tree input dimension."
    
    println("\n⚡ Starting Sparse Tree Construction...")
    println("   - Points to process: $n_points")
    println("   - Threads available: $(Threads.nthreads())")
    
    # --- PHASE 1: Parallel Forward Pass ---
    # We pre-allocate an array to store the paths. 
    paths = Vector{Vector{Vector{Int}}}(undef, n_points)
    
    p1 = Progress(n_points; desc="Computing Paths (Parallel): ")
    Threads.@threads for i in 1:n_points
        paths[i] = compute_activation_path(tree, points[:, i])
        next!(p1)
    end
    
    # --- PHASE 2: Sequential Tree Assembly ---
    # Sequential assembly prevents race conditions on parent.children without needing locks.
    p2 = Progress(n_points; desc="Assembling Tree (Sequential): ")
    for i in 1:n_points
        insert_path!(tree, points[:, i], paths[i])
        next!(p2)
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