module Utils

export find_hyperplanes

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

end # module
