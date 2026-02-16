module Utils

using ..Regions
using Polyhedra
using CDDLib
using LinearAlgebra

export find_hyperplanes, get_region_volume, analyze_region

function find_hyperplanes(state_dict::Dict{String, Any})
    # FIX: Support both legacy "weight" (PyTorch) and new "W_" (Julia Save) formats
    keys_weights = filter(k -> occursin("weight", k) || startswith(k, "W_"), collect(keys(state_dict)))

    function extract_layer_idx(k)
        m = match(r"(\d+)", k)
        return m === nothing ? -1 : parse(Int, m.captures[1])
    end

    sorted_keys = sort(keys_weights, by=extract_layer_idx)

    weights = Matrix{Float64}[]
    biases = Vector{Float64}[]

    for k_w in sorted_keys
        # Determine corresponding bias key
        if occursin("weight", k_w)
            k_b = replace(k_w, "weight" => "bias")
        else
            k_b = replace(k_w, "W_" => "b_")
        end
        
        if !haskey(state_dict, k_b)
             error("Bias key '$k_b' not found for weight '$k_w'")
        end

        W = convert(Matrix{Float64}, state_dict[k_w])
        b = convert(Vector{Float64}, state_dict[k_b])
        push!(weights, W)
        push!(biases, b)
    end
    return weights, biases
end


# ==============================================================================
# 1. VOLUME CALCULATION & BOUNDING
# ==============================================================================

function get_region_volume(region::Region; bound::Union{Float64, Nothing}=nothing)
    # 1. Get Region Constraints
    A, b = get_path_inequalities(region)
    dim = size(A, 2)
    
    # 2. Add Bounding Box Constraints if requested
    if !isnothing(bound)
        I_mat = Matrix{Float64}(I, dim, dim)
        A_box = vcat(I_mat, -I_mat)
        b_box = fill(bound, 2 * dim)
        A = vcat(A, A_box)
        b = vcat(b, b_box)
    end

    # 3. Construct Polyhedron
    h = hrep(A, b)
    poly = polyhedron(h, CDDLib.Library())

    # 4. Check Unboundedness
    if isnothing(bound)
        try
            # Convert to V-representation (vertices + rays)
            vr = vrep(poly)
            # If there are any rays, the volume is infinite
            if !isempty(rays(vr))
                return Inf
            end
        catch
            # If conversion fails, assume complex/degenerate or handle downstream
        end
    end

    # 5. Compute Volume
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

function analyze_region(region; bound::Union{Float64, Nothing}=nothing)
    # 1. Get Path Constraints
    # FIX: Use function imported from Regions, not Geobin global
    A, b = get_path_inequalities(region)
    dim = size(A, 2)
    
    # 2. Apply Bounding Box (if requested)
    if !isnothing(bound)
        A_box = vcat(Matrix{Float64}(I, dim, dim), -Matrix{Float64}(I, dim, dim))
        b_box = fill(bound, 2 * dim)
        A = vcat(A, A_box)
        b = vcat(b, b_box)
    end

    # 3. Construct Polyhedron
    h = hrep(A, b)
    poly = polyhedron(h, CDDLib.Library())

    # 4. Check for Emptiness
    removehredundancy!(poly)
    if isempty(poly)
        return (0.0, true) # Empty sets are bounded with volume 0
    end

    # 5. Handle Unboundedness (Logic Split)
    try
        # Convert to V-representation (Vertices + Rays)
        vr = vrep(poly)
        
        # Check for Rays (Infinite directions)
        if !isempty(rays(vr))
            if !isnothing(bound)
                @warn "Region has rays despite bounding box! Numerical instability likely."
            end
            return (Inf, false)
        end
        
        # 6. Compute Volume
        vol = volume(poly)
        return (vol, true)

    catch e
        @warn "Volume computation failed for Region $(region.id): $e"
        return (NaN, false) 
    end
end

end # module