module Regions

export Region, add_child!, get_children

mutable struct Region
    layer_number::Int
    qlw::BitVector             # Highly compressed activation signature
    active_indices::Vector{Int32} # Stored ONLY for exact tree, empty for sparse
    x::Vector{Float64}         # Feasible interior point
    bounded::Bool
    volume_ex::Float64
    volume_es::Float64
    
    parent::Union{Region, Nothing}
    children::Vector{Region}

    # Root constructor
    function Region(; input_dim::Int) 
        this = new()
        this.layer_number = 0
        this.qlw = BitVector()
        this.active_indices = Int32[]
        this.x = zeros(input_dim)
        this.bounded = false
        this.volume_ex = Inf
        this.volume_es = Inf
        this.parent = nothing
        this.children = Region[]
        return this
    end

    # Child constructor
    function Region(activation::BitVector, layer::Int)
        this = new()
        this.layer_number = layer
        this.qlw = activation
        this.active_indices = Int32[]
        this.volume_ex = 0.0
        this.volume_es = 0.0
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

end # module