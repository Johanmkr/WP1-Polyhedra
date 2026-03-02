module Construction

using ..Regions
using ..Trees
using ..Geometry
using LinearAlgebra
using JuMP
using HiGHS

export construct_tree!

function construct_tree!(tree::Tree; verbose::Bool=false)
    A_root = Matrix{Float64}(I, tree.input_dim, tree.input_dim)
    c_root = zeros(Float64, tree.input_dim)
    D_root = Matrix{Float64}(undef, 0, tree.input_dim)
    g_root = Float64[]
    
    # DFS can run sequentially or in parallel batches, but sequentially is robust and memory-flat.
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    verbose && println("🚀 Starting Exact Tree Construction (DFS)...")
    _build_dfs!(tree, tree.root, 1, A_root, c_root, D_root, g_root, model)
    verbose && println("✅ Exact Tree Construction Complete.")
    return tree
end

function _build_dfs!(tree::Tree, parent::Region, layer_idx::Int, 
                     A_prev::Matrix{Float64}, c_prev::Vector{Float64}, 
                     D_prev::Matrix{Float64}, g_prev::Vector{Float64}, model::Model)
                     
    if layer_idx > tree.L
        return # Reached maximum depth
    end
    
    W = tree.weights[layer_idx]
    b = tree.biases[layer_idx]
    
    # Calculate base W_hat, b_hat once for this parent node
    W_hat = W * A_prev
    b_hat = W * c_prev + b
    
    # Start with the feasible point from the parent
    z_start = W_hat * parent.x + b_hat
    q_start = BitVector(z_start .> 0)
    
    queue = [q_start]
    visited = Set{BitVector}([q_start])
    
    while !isempty(queue)
        q_curr = popfirst!(queue)
        
        # Calculate Ephemeral D, g for this specific child candidate
        s_vec = -2.0 .* q_curr .+ 1.0
        D_local = s_vec .* W_hat
        g_local = -(s_vec .* b_hat)
        
        A_next = q_curr .* W_hat
        c_next = q_curr .* b_hat
        
        # 1. Check Feasibility & Extract Active Indices
        active_indices, is_bounded, vol = find_active_indices_exact(D_local, g_local, D_prev, g_prev)
        feasible_point = get_feasible_point([D_prev; D_local], [g_prev; g_local]; model=model)
        
        if !isnothing(feasible_point)
            child = Region(q_curr, layer_idx)
            child.x = feasible_point
            child.active_indices = Int32.(active_indices)
            child.bounded = is_bounded
            child.volume_ex = vol
            
            add_child!(parent, child)
            
            # 2. RECURSE DEEPER (Depth-First)
            _build_dfs!(tree, child, layer_idx + 1, 
                        A_next, c_next, 
                        [D_prev; D_local], [g_prev; g_local], 
                        model)
            
            # 3. Queue neighbors (facet-flipping) based on active indices
            for idx in active_indices
                q_neighbor = copy(q_curr)
                q_neighbor[idx] = !q_neighbor[idx]
                if !(q_neighbor in visited)
                    push!(visited, q_neighbor)
                    push!(queue, q_neighbor)
                end
            end
        end
    end
end

end # module