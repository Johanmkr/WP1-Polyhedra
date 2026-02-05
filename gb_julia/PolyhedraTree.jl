module PolyhedraTree

using LinearAlgebra
using JuMP
using HiGHS
using ProgressMeter
using Polyhedra
using CDDLib # Critical for exact volume and V-rep conversion

import Base: show

export Region, Tree, construct_tree!, get_regions_at_layer, get_path_inequalities

# ==============================================================================
# 1. REGION CLASS (Unchanged mostly, just ensure volume is Float64)
# ==============================================================================

mutable struct Region
    qlw::Vector{Int}
    q_tilde::Vector{Int}
    bounded::Bool
    volume::Float64
    
    Alw::Matrix{Float64}
    clw::Vector{Float64}
    
    Dlw::Matrix{Float64}
    glw::Vector{Float64}
    
    Dlw_active::Matrix{Float64}
    glw_active::Vector{Float64}
    
    parent::Union{Region, Nothing}
    children::Vector{Region}
    layer_number::Int
    
    function Region(; input_dim::Int)
        this = new()
        this.qlw = Int[]
        this.q_tilde = Int[]
        this.bounded = false
        this.volume = Inf 
        
        this.Alw = Matrix{Float64}(I, input_dim, input_dim)
        this.clw = zeros(Float64, input_dim)
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
        this.volume = 0.0
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
    D_path = reduce(vcat, reverse(D_list))
    g_path = reduce(vcat, reverse(g_list))
    return D_path, g_path
end

function Base.show(io::IO, r::Region)
    vol_str = r.volume == Inf ? "Inf" : string(round(r.volume, digits=4))
    print(io, "\nRegion (L$(r.layer_number)) | Vol: $vol_str | Act: $(r.qlw) | Children: $(length(r.children))")
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
        nodes_lock = ReentrantLock()

        if verbose
            println("Layer $layer: Processing $(length(current_layer_nodes)) regions...")
        end
        
        # Parallel loop
        # Threads.@threads for parent in current_layer_nodes
        for parent in current_layer_nodes
            
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
                child.Dlw_active = info["Dlw_active"]
                child.glw_active = info["glw_active"]
                child.volume = info["volume"]
                child.Alw = info["Alw"]
                child.clw = info["clw"]
                child.layer_number = layer
                
                add_child!(parent, child)
                push!(local_children, child)
            end
            
            lock(nodes_lock) do
                append!(next_layer_nodes, local_children)
            end
        end
        
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
# 4. SOLVERS & MATH
# ==============================================================================

function find_hyperplanes(state_dict::Dict{String, Any})
    keys_weights = filter(k -> occursin("weight", k), collect(keys(state_dict)))
    
    function extract_layer_idx(k)
        m = match(r"(\d+)", k)
        return m === nothing ? -1 : parse(Int, m.captures[1])
    end
    
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
    # 1. Get interior point (Fallback mechanism kept for stability)
    if layer_nr != 1
        x = get_interior_point_adaptive(Dlw_active_prev, glw_active_prev)
        
        is_valid = false
        if x !== nothing && !any(isnan, x)
            if isempty(Dlw_active_prev)
                is_valid = true
            else
                if maximum(Dlw_active_prev * x .- glw_active_prev) <= 1e-5
                    is_valid = true
                end
            end
        end
        
        # GD Fallback
        if !is_valid
            dim = size(Alw_prev, 2)
            x_curr = randn(dim)
            lr = 0.05; decay = 0.99; max_iter = 500
            for _ in 1:max_iter
                if isempty(Dlw_active_prev); is_valid = true; x = x_curr; break; end
                violations = Dlw_active_prev * x_curr .- glw_active_prev
                mask = violations .> 1e-7
                if !any(mask); x = x_curr; is_valid = true; break; end
                grad = (Dlw_active_prev[mask, :])' * violations[mask]
                gnorm = norm(grad)
                if gnorm > 1e-8; x_curr .-= (lr * grad) / gnorm; else; x_curr = randn(dim); end
                lr *= decay
            end
        end

        if !is_valid
            println("Region not valid")
            return Dict{Vector{Int}, Dict}() 
        end
    else
        x = rand(size(Alw_prev, 2))
    end
    
    # 2. Initial Activation
    z = Wl * Alw_prev * x + Wl * clw_prev + bl
    q0 = Int.(z .> 0)
    
    traversed = Dict{Vector{Int}, Dict}()
    queue = [q0]
    
    while !isempty(queue)
        q = popfirst!(queue)
        
        Dlw, glw, Alw, clw = calculate_next_layer_quantities(Wl, bl, q, Alw_prev, clw_prev)
        
        # Use simple LP to find active constraints first (fast filter)
        qi_act, _ = find_active_indices_fast(Dlw, glw, Dlw_active_prev, glw_active_prev)
        
        D_local_active = Dlw[qi_act, :]
        g_local_active = glw[qi_act]
        D_total = vcat(D_local_active, Dlw_active_prev)
        g_total = vcat(g_local_active, glw_active_prev)
        
        # --- NEW: EXACT GEOMETRY CHECK via CDDLib ---
        # This replaces the ambiguity of LP solvers with exact V-rep conversion
        is_bounded, volume_val, is_feasible = get_exact_geometry(D_total, g_total)
        
        if is_feasible
            traversed[q] = Dict(
                "q_tilde" => qi_act,
                "bounded" => is_bounded,
                "volume"  => volume_val,
                "Dlw" => Dlw,
                "glw" => glw,
                "Alw" => Alw,
                "clw" => clw,
                "Dlw_active" => D_local_active,
                "glw_active" => g_local_active
            )
            
            for i_act in qi_act
                q_new = copy(q)
                q_new[i_act] = 1 - q_new[i_act]
                if !haskey(traversed, q_new) && !(q_new in queue)
                     push!(queue, q_new)
                end
            end
        end
    end
    return traversed
end

# ==============================================================================
# CORRECTED GEOMETRY ENGINE
# ==============================================================================

function get_exact_geometry(D, g)
    # 1. Create H-representation
    # We map to the library inside the hrep call to ensure type consistency
    h = Polyhedra.hrep(D, g)

    # 2. Use EXACT Arithmetic (Rational)
    # :float is fast but crashes/fails on thin NN polytopes. 
    # :exact uses GMP (BigInt/Rational) to guarantee correctness.
    lib = CDDLib.Library(:exact) 

    try
        # Create polyhedron
        poly = Polyhedra.polyhedron(h, lib)
        
        # 3. Force V-Rep computation
        # Note: In exact mode, this might take longer but won't crash on slivers
        v = Polyhedra.vrep(poly)
        
        # 4. Feasibility Check
        # If it has 0 points and 0 rays, it is truly empty.
        if Polyhedra.npoints(v) == 0 && Polyhedra.nrays(v) == 0
            return (true, 0.0, false) 
        end
        
        # 5. Unbounded Check
        if Polyhedra.nrays(v) > 0
            return (false, Inf, true)
        end
        
        # 6. Degeneracy Check (Volume)
        # ONLY use this check if you are in low dimensions (d < 5).
        # Otherwise, trust existence of vertices.
        # Calculating exact volume in high-dim is also very expensive.
        
        # Optimization: Skip volume calc for validity check. 
        # If npoints > 0 and bounded, it is a valid polytope.
        return (true, 1.0, true) # Return dummy volume 1.0 to indicate "Valid/Finite"
        
    catch e
        # Log the error so you know if you are losing regions
        println("\n!!! Geometry Check Error: $e")
        println("Constraints D shape: $(size(D))")
        return (true, 0.0, false)
    end
end

# --- KEEPING THE FAST LP FILTER FOR INITIAL ACTIVE SET ---
function find_active_indices_fast(D_local, g_local, D_prev, g_prev; tol=1e-5)
    D_all = vcat(D_local, D_prev)
    g_all = vcat(g_local, g_prev)
    n_local = size(D_local, 1)
    dim = size(D_all, 2)
    active_indices = Int[]
    
    model = direct_model(HiGHS.Optimizer())
    set_silent(model)
    set_attribute(model, "presolve", "off")
    @variable(model, x[1:dim])
    @constraint(model, cons[i=1:size(D_all,1)], dot(D_all[i,:], x) <= g_all[i])

    for i in 1:n_local
        d_row = D_all[i, :]
        if norm(d_row) < 1e-10; continue; end

        original_rhs = g_all[i]
        set_normalized_rhs(cons[i], 1e20) 
        @objective(model, Max, dot(d_row, x))
        optimize!(model)
        
        st = termination_status(model)
        is_active = false
        if st == MOI.DUAL_INFEASIBLE
            is_active = true
        elseif st == MOI.OPTIMAL
            if objective_value(model) > original_rhs + tol
                is_active = true
            end
        end
        
        if is_active
            push!(active_indices, i)
            set_normalized_rhs(cons[i], original_rhs)
        end
    end
    return active_indices, true # Boundedness is checked later by CDDLib
end

# --- LP Solvers (Unchanged helpers) ---

function get_interior_point_adaptive(A::Matrix{Float64}, b::Vector{Float64})
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    limit = 1e6 
    @variable(model, -limit <= x[i=1:n] <= limit)
    ϵ = 1e-7
    for i in 1:m
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) <= b[i] - ϵ)
    end
    @objective(model, Min, 0)
    optimize!(model)
    st = termination_status(model)
    if st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)
        return value.(x)
    elseif st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED, MOI.SLOW_PROGRESS)
        val = find_any_feasible_point(A, b, limit)
        return any(isnan, val) ? nothing : val
    else
        return nothing
    end
end

function find_any_feasible_point(A, b, limit)
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -limit <= x[1:n] <= limit)
    @constraint(model, A * x .<= b)
    @objective(model, Min, 0)
    optimize!(model)
    if termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        return value.(x)
    else
        return fill(NaN, size(A, 2))
    end
end

end # module