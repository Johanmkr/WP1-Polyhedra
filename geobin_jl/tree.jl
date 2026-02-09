module Trees

using ..Regions
using ..Utils

export Tree, get_regions_at_layer

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

end # module
