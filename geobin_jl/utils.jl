module Utils

using ..Regions
using Polyhedra
using CDDLib
using LinearAlgebra

export find_hyperplanes, get_region_volume, analyze_region
export get_activation_path, compute_path_geometry

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

# ==============================================================================
# EPHEMERAL GEOMETRY ENGINE
# ==============================================================================

"""
Traces up the tree to get the sequence of activations from Layer 1 to this region.
"""
function get_activation_path(r::Region)
    path = BitVector[]
    curr = r
    while curr.parent !== nothing && curr.layer_number > 0
        push!(path, curr.qlw)
        curr = curr.parent
    end
    return reverse(path)
end

"""
Dynamically recalculates the path inequalities D*x <= g 
for a given activation path, consuming virtually zero permanent memory.
"""
function compute_path_geometry(weights::Vector{Matrix{Float64}}, biases::Vector{Vector{Float64}}, q_path::Vector{BitVector}; active_indices=Int32[])
    input_dim = size(weights[1], 2)
    
    # FIX: Handle the root node (empty path) safely
    if isempty(q_path)
        return Matrix{Float64}(undef, 0, input_dim), Float64[]
    end
    
    A_curr = Matrix{Float64}(I, input_dim, input_dim)
    c_curr = zeros(Float64, input_dim)
    
    D_list = Matrix{Float64}[]
    g_list = Vector{Float64}[]
    
    for l in 1:length(q_path)
        W = weights[l]
        b = biases[l]
        q = q_path[l]
        
        W_hat = W * A_curr
        b_hat = W * c_curr + b
        
        # s_vec is -1 if q=0, and 1 if q=1
        s_vec = -2.0 .* q .+ 1.0
        
        D_local = s_vec .* W_hat
        g_local = -(s_vec .* b_hat)
        
        push!(D_list, D_local)
        push!(g_list, g_local)
        
        A_curr = q .* W_hat
        c_curr = q .* b_hat
    end
    
    D_full = reduce(vcat, D_list)
    g_full = reduce(vcat, g_list)
    
    # If active_indices are provided, extract only the minimal set
    if !isempty(active_indices)
        return D_full[active_indices, :], g_full[active_indices]
    end
    
    return D_full, g_full
end

# ==============================================================================
# VOLUME CALCULATION 
# ==============================================================================

function get_region_volume(region::Region, weights::Vector{Matrix{Float64}}, biases::Vector{Vector{Float64}}; bound::Union{Float64, Nothing}=nothing)
    q_path = get_activation_path(region)
    # CDDLib handles redundancy natively, so we pass the full D and g
    A, b = compute_path_geometry(weights, biases, q_path)
    dim = size(A, 2)
    
    if !isnothing(bound)
        I_mat = Matrix{Float64}(I, dim, dim)
        A_box = vcat(I_mat, -I_mat)
        b_box = fill(bound, 2 * dim)
        A = vcat(A, A_box)
        b = vcat(b, b_box)
    end

    h = hrep(A, b)
    poly = polyhedron(h, CDDLib.Library())

    if isnothing(bound)
        try
            vr = vrep(poly)
            if !isempty(rays(vr))
                return Inf
            end
        catch
        end
    end

    try
        removevredundancy!(poly)
        if isempty(poly)
            return 0.0
        end
        return volume(poly)
    catch e
        return 0.0
    end
end

end # module