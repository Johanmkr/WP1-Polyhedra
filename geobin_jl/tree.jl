module Trees

using ..Regions
using ..Utils
using Printf

export Tree, get_regions_at_layer, print_tree_summary

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

"""
    print_tree_summary(tree::Tree)

Prints a comprehensive table showing:
1. Neural Network Architecture (Neurons per layer)
2. Partition Complexity (Regions per layer)
3. Branching Factor (Average number of children per parent)
"""
function print_tree_summary(tree::Tree)
    println("\n" * "="^65)
    println("🌲 Geometric Tree Summary")
    println("="^65)
    
    @printf("• %-20s : %d\n", "Input Dimension", tree.input_dim)
    @printf("• %-20s : %d\n", "Depth (Layers)", tree.L)
    
    println("\n" * "-"^65)
    println(" Layer |  Neurons  |  Regions  |  Growth (Avg Children) ")
    println("-"^65)
    
    total_regions = 0
    prev_count = 1 # Start with Root count
    
    for l in 0:tree.L
        regions = get_regions_at_layer(tree, l)
        count = length(regions)
        total_regions += count
        
        # 1. Architecture Info
        if l == 0
            arch_str = "Input"
        else
            # Weights[l] corresponds to the transition to layer l
            # Size is (n_neurons, n_prev_neurons)
            n_neurons = size(tree.weights[l], 1)
            arch_str = "$n_neurons"
        end
        
        # 2. Branching Factor (Growth)
        # Avoid division by zero if a layer is empty
        if l == 0
            growth_str = "-"
        elseif prev_count > 0
            growth_val = count / prev_count
            growth_str = @sprintf("%.2fx", growth_val)
        else
            growth_str = "0.00x"
        end

        # Print Row
        @printf("   %2d  |  %-7s  |  %7d  |  %15s\n", l, arch_str, count, growth_str)
        
        prev_count = count
    end
    
    println("-"^65)
    println("   TOTAL NODES   |  $total_regions")
    println("="^65 * "\n")
end

# Optional: Cleaner REPL output for the object itself
function Base.show(io::IO, t::Tree)
    print(io, "Tree(depth=$(t.L), input_dim=$(t.input_dim), total_nodes=$(length(get_children(t.root)) > 0 ? "..." : "1"))")
end

end # module