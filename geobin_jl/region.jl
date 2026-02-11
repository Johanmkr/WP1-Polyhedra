module Regions

using LinearAlgebra

export Region, add_child!, get_children, get_path_inequalities

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
    x::Vector{Float64}

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
        this.x = rand(input_dim)
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

end # module
