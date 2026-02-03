module PolyhedraTree

using LinearAlgebra
using JuMP
using HiGHS
using ProgressMeter
import Base: show

export Region, Tree, construct_tree!, get_regions_at_layer, get_path_inequalities

# ==============================================================================
# 1. REGION CLASS
# ==============================================================================

mutable struct Region
    # Region attributes
    qlw::Vector{Int}                # Activation pattern
    q_tilde::Vector{Int}            # Active bits indices
    bounded::Bool
    
    # Inequalities and projections
    Alw::Matrix{Float64}            # Slope projection matrix
    clw::Vector{Float64}            # Intercept projection matrix
    
    Dlw::Matrix{Float64}            # Slopes of inequalities
    glw::Vector{Float64}            # Intercept of inequalities 
    
    Dlw_active::Matrix{Float64}     # Active slopes
    glw_active::Vector{Float64}     # Active intercepts
    
    # Tree attributes
    parent::Union{Region, Nothing}
    children::Vector{Region}
    
    # Utility attributes
    layer_number::Int
    
    # -- Constructors --
    function Region(; input_dim::Int)
        this = new()
        this.qlw = Int[]
        this.q_tilde = Int[]
        this.bounded = true
        
        # Identity projection for root
        this.Alw = Matrix{Float64}(I, input_dim, input_dim)
        this.clw = zeros(Float64, input_dim)
        
        # Generate hypercube inequalities (|x_i| <= 1)
        # REMOVED: Python implementation uses unbounded root region.
        # Matching Python behavior to find all regions and avoid artificial overlaps.
        this.Dlw = Matrix{Float64}(undef, 0, input_dim)
        this.glw = Float64[]
        
        this.Dlw_active = this.Dlw
        this.glw_active = this.glw
        
        this.parent = nothing
        this.children = Region[]
        this.layer_number = 0
        return this
    end

    function Region(activation::Vector{Int})
        this = new()
        this.qlw = activation
        this.children = Region[]
        return this
    end
end

function add_child!(parent::Region, child::Region)
    child.parent = parent
    push!(parent.children, child)
end

function get_children(r::Region)
    return r.children
end

# Matches Python: get_path_inequalities(self)
function get_path_inequalities(r::Region)
    D_list = Matrix{Float64}[]
    g_list = Vector{Float64}[]
    
    node = r
    while node.parent !== nothing
        push!(D_list, node.Dlw_active)
        push!(g_list, node.glw_active)
        node = node.parent
    end
    
    if isempty(D_list)
        return Matrix{Float64}(undef, 0, 0), Float64[]
    end
    
    # Reverse to get Root -> Leaf order
    D_path = reduce(vcat, reverse(D_list))
    g_path = reduce(vcat, reverse(g_list))
    
    return D_path, g_path
end

function Base.show(io::IO, r::Region)
    dim = isdefined(r, :Alw) ? size(r.Alw, 2) : "N/A"
    print(io, "\nRegion (L$(r.layer_number)) | Act: $(r.qlw) | Children: $(length(r.children))")
end

# ==============================================================================
# 2. TREE CLASS
# ==============================================================================

mutable struct Tree
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    input_dim::Int
    L::Int
    root::Region

    function Tree(state_dict::Dict{String, Any})
        weights, biases = find_hyperplanes(state_dict)
        input_dim = size(weights[1], 2)
        L = length(weights)
        root = Region(input_dim=input_dim)
        new(weights, biases, input_dim, L, root)
    end
end

# ==============================================================================
# 3. CONSTRUCTION LOGIC
# ==============================================================================

function construct_tree!(tree::Tree; verbose::Bool=false)
    current_layer_nodes = [tree.root]
    
    for i in 1:tree.L
        Wl = tree.weights[i]
        bl = tree.biases[i]
        layer = i 
        
        next_layer_nodes = Region[]
        
        if verbose
            p = Progress(length(current_layer_nodes); dt=0.5, desc="Layer $layer: ")
        end
        
        # Using a lock for thread safety if multi-threaded
        list_lock = ReentrantLock()
        
        # You can use Threads.@threads here if BLAS threads are set to 1
        for parent in current_layer_nodes
            
            # Core solver step
            new_nodes_info = find_next_layer_region_info(
                parent.Dlw_active, parent.glw_active, 
                parent.Alw, parent.clw, 
                Wl, bl, layer
            )
            
            local_children = Region[]
            
            for (act, info) in new_nodes_info
                child = Region(act)
                child.q_tilde = info["q_tilde"]
                child.bounded = info["bounded"]
                child.Dlw = info["Dlw"]
                child.glw = info["glw"]
                
                # Slicing active constraints
                child.Dlw_active = info["Dlw"][info["q_tilde"], :]
                child.glw_active = info["glw"][info["q_tilde"]]
                
                child.Alw = info["Alw"]
                child.clw = info["clw"]
                child.layer_number = layer
                
                add_child!(parent, child)
                push!(local_children, child)
            end
            
            lock(list_lock) do
                append!(next_layer_nodes, local_children)
                if verbose; next!(p); end
            end
        end
        
        if verbose; finish!(p); end
        
        current_layer_nodes = next_layer_nodes
        if isempty(current_layer_nodes)
            break
        end
    end
end

function get_regions_at_layer(tree::Tree, layer::Int)
    regions = Region[]
    queue = [tree.root]
    
    while !isempty(queue)
        current_region = popfirst!(queue)
        if current_region.layer_number == layer
            push!(regions, current_region)
        elseif current_region.layer_number < layer
            append!(queue, current_region.children)
        end
    end
    return regions
end

# ==============================================================================
# 4. SOLVERS & MATH (Strictly Matching Python Logic)
# ==============================================================================

function find_hyperplanes(state_dict::Dict{String, Any})
    # --- FIX: ROBUST SORTING ---
    # Python uses insertion order or string keys. 
    # We must extract the integer layer index to ensure 10 comes after 2.
    
    # 1. Filter weight keys
    keys_weights = filter(k -> occursin("weight", k), collect(keys(state_dict)))
    
    # 2. Extract integer index from key (e.g. "layers.0.weight" -> 0)
    # Assumes format contains digits we can sort by.
    function extract_layer_idx(k)
        m = match(r"(\d+)", k)
        return m === nothing ? -1 : parse(Int, m.captures[1])
    end
    
    # 3. Sort based on the integer index
    sorted_keys = sort(keys_weights, by=extract_layer_idx)
    
    weights = Matrix{Float64}[]
    biases = Vector{Float64}[]
    
    for k_w in sorted_keys
        k_b = replace(k_w, "weight" => "bias")
        W = convert(Matrix{Float64}, state_dict[k_w])
        b = convert(Vector{Float64}, state_dict[k_b])
        push!(weights, W)
        push!(biases, b)
    end
    
    return weights, biases
end

function calculate_next_layer_quantities(Wl, bl, qlw, Alw_prev, clw_prev)
    Wl_hat = Wl * Alw_prev
    bl_hat = Wl * clw_prev + bl
    
    s_vec = -2.0 .* qlw .+ 1.0
    
    Dlw = s_vec .* Wl_hat
    glw = -(s_vec .* bl_hat)
    
    Alw = qlw .* Wl_hat
    clw = qlw .* bl_hat
    
    return Dlw, glw, Alw, clw
end

function find_next_layer_region_info(Dlw_active_prev, glw_active_prev, Alw_prev, clw_prev, Wl, bl, layer_nr)
    # 1. Get interior point
    if layer_nr != 1
        x = get_interior_point_adaptive(Dlw_active_prev, glw_active_prev)
    else
        # Root region is unbounded. Python uses random point in [0, 1]^d.
        # Matching Python behavior.
        x = rand(size(Alw_prev, 2))
    end
    
    # 2. Initial Activation
    z = Wl * Alw_prev * x + Wl * clw_prev + bl
    q0 = Int.(z .> 0)
    
    # 3. BFS
    traversed = Dict{Vector{Int}, Dict}()
    queue = [q0]
    
    while !isempty(queue)
        q = popfirst!(queue)
        
        Dlw, glw, Alw, clw = calculate_next_layer_quantities(Wl, bl, q, Alw_prev, clw_prev)
        
        qi_act, is_bounded = find_active_indices(Dlw, glw, Dlw_active_prev, glw_active_prev)
        
        for i_act in qi_act
            q_new = copy(q)
            q_new[i_act] = 1 - q_new[i_act]
            
            if !haskey(traversed, q_new)
                traversed[q_new] = Dict() # Placeholder
                push!(queue, q_new)
            end
        end
        
        traversed[q] = Dict(
            "q_tilde" => qi_act,
            "bounded" => is_bounded,
            "Dlw" => Dlw,
            "glw" => glw,
            "Alw" => Alw,
            "clw" => clw
        )
    end
    return traversed
end

# --- LP Solvers ---

# ==============================================================================
# UPDATED SOLVER FUNCTIONS
# Replace the existing functions in PolyhedraTree.jl with these
# ==============================================================================

function get_interior_point_adaptive(A::Matrix{Float64}, b::Vector{Float64}; initial_slack=0.1, min_threshold=1e-12)
    m, n = size(A)
    model = direct_model(HiGHS.Optimizer())
    set_silent(model)
    
    # Strict feasibility to prevent "ghost" regions
    set_attribute(model, "primal_feasibility_tolerance", 1e-9)
    set_attribute(model, "dual_feasibility_tolerance", 1e-9)
    # Disable presolve to prevent it from discarding 'tiny' valid features
    set_attribute(model, "presolve", "off")

    @variable(model, x[1:n])
    
    norms = [norm(A[i, :]) for i in 1:m]
    current_slack = initial_slack
    
    # 1. Try finding a deep interior point
    while current_slack >= min_threshold
        @constraint(model, [i=1:m], dot(A[i,:], x) <= b[i] - (current_slack * norms[i]))
        optimize!(model)
        
        if termination_status(model) == MOI.OPTIMAL
            return value.(x)
        end
        
        # Retry with smaller slack
        empty!(model) 
        @variable(model, x[1:n])
        current_slack /= 10.0
    end
    
    # 2. Final attempt: Zero slack (on the boundary)
    @constraint(model, [i=1:m], dot(A[i,:], x) <= b[i])
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(x)
    else
        error("Polytope empty/infeasible.")
    end
end

function find_active_indices(D_local, g_local, D_prev, g_prev; tol=1e-7)
    # 1. Merge constraints
    D_all = vcat(D_local, D_prev)
    g_all = vcat(g_local, g_prev)
    
    n_local = size(D_local, 1)
    n_total, dim = size(D_all)
    
    active = Int[]
    is_bounded = true
    
    # 2. Setup Solver
    model = direct_model(HiGHS.Optimizer())
    set_silent(model)
    
    # Strict settings to prevent "ghost" regions
    set_attribute(model, "primal_feasibility_tolerance", 1e-9)
    set_attribute(model, "dual_feasibility_tolerance", 1e-9)
    set_attribute(model, "presolve", "off") # Important: Don't let solver delete "redundant" tiny rows

    @variable(model, x[1:dim])
    @constraint(model, cons[i=1:n_total], dot(D_all[i,:], x) <= g_all[i])
    
    # 3. Iterate local constraints
    for i in 1:n_local
        d_row = D_all[i, :]
        row_norm = norm(d_row)
        
        # Skip zero-rows (shouldn't happen in valid NNs)
        if row_norm < 1e-12
            continue
        end

        # Save original RHS
        original_rhs = g_all[i]
        
        # RELAX: Remove the wall
        set_normalized_rhs(cons[i], Inf)
        
        # OBJECTIVE: Maximize in direction of normal
        @objective(model, Max, dot(d_row, x))
        
        optimize!(model)
        st = termination_status(model)
        
        # 4. Check Activity with NORMALIZATION
        if st == MOI.OPTIMAL
            val = objective_value(model)
            
            # Calculate geometric distance violated
            # dist = (val - rhs) / ||d||
            geometric_violation = (val - original_rhs) / row_norm
            
            if geometric_violation > tol
                push!(active, i)
            end
            
        elseif st == MOI.DUAL_INFEASIBLE
            # Unbounded
            push!(active, i)
            is_bounded = false
        end
        
        # RESTORE
        set_normalized_rhs(cons[i], original_rhs)
    end
    
    return active, is_bounded
end

end # module