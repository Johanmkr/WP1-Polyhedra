module SparseConstruction

using ..Regions
using ..Trees
using LinearAlgebra
using ProgressMeter
using Base.Threads: ReentrantLock

export construct_tree_sparse!, construct_tree_sparse_mc!

function compute_activation_path(tree::Tree, x0::AbstractVector{Float32})
    q_path = Vector{BitVector}(undef, tree.L)
    a = x0
    
    for l in 1:tree.L
        z = tree.weights[l] * a + tree.biases[l]
        q = BitVector(z .> 0)
        q_path[l] = q
        a = q .* z 
    end
    
    return q_path
end

function insert_path!(tree::Tree, x0::AbstractVector{Float32}, q_path::Vector{BitVector}, root_lock::ReentrantLock)
    current_node = tree.root
    
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
        
        if !isnothing(existing_child)
            current_node = existing_child
        else
            # 2. Path doesn't exist, create a new lightweight Region
            new_node = Region(q, l)
            new_node.x = x0 
            
            # 3. Safely attach the node
            if l == 1
                lock(root_lock) do
                    add_child!(current_node, new_node)
                end
            else
                add_child!(current_node, new_node)
            end
            current_node = new_node
        end
    end
end

function construct_tree_sparse!(tree::Tree, points::Matrix{Float32})
    dim, n_points = size(points)
    @assert dim == tree.input_dim "Point dimensions do not match tree input dimension."
    println("\n⚡ Starting Sparse Tree Construction...")
    
    paths = Vector{Vector{BitVector}}(undef, n_points)
    
    p1 = Progress(n_points; desc="Computing Paths (Parallel): ")
    Threads.@threads for i in 1:n_points
        paths[i] = compute_activation_path(tree, @view(points[:, i]))
        next!(p1)
    end
    
    # Assembly
    root_lock = ReentrantLock() 
    p2 = Progress(n_points; desc="Assembling Subtrees: ")
    
    # We can still batch by Layer 1 branches, but even a simple loop is extremely fast now
    for i in 1:n_points
        insert_path!(tree, @view(points[:, i]), paths[i], root_lock)
        next!(p2)
    end
    
    println("✅ Sparse tree construction complete.")
    return tree
end

function construct_tree_sparse_mc!(tree::Tree, num_points::Int, bound::Float64)
    points = Float32.((rand(tree.input_dim, num_points) .- 0.5) .* (2 * bound))
    return construct_tree_sparse!(tree, points)
end

end # module