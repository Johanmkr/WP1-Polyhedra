module SaveTree

using HDF5
using ..Regions
using ..Trees

export read_state_dict_from_h5, save_tree_to_h5

"""
    read_state_dict_from_h5(ep_group::HDF5.Group)

Reads weights and biases from an HDF5 group representing a single epoch.
Automatically transposes PyTorch (Out, In) weight matrices to Julia (In, Out) format.
"""
function read_state_dict_from_h5(ep_group::HDF5.Group)
    state_dict = Dict{String, Any}()
    for k in keys(ep_group)
        if occursin("weight", k)
            data = read(ep_group[k])
            # Transpose PyTorch weights (Out, In) -> Julia (In, Out)
            if ndims(data) == 2
                data = permutedims(data, (2, 1))
            end
            state_dict[k] = data
        elseif occursin("bias", k)
            data = read(ep_group[k])
            state_dict[k] = data
        end
    end
    return state_dict
end

"""
    save_tree_to_h5(g::HDF5.Group, tree::Tree)

Linearizes a PolyhedraTree and saves it into the given HDF5 group using 
dimensionality-aware compression and chunking. Overwrites existing tree data if present.
"""
function save_tree_to_h5(g::HDF5.Group, tree::Tree)
    node_list = Region[]
    queue = [tree.root]
    region_to_idx = Dict{UInt64, Int}()
    
    # 1. Linearize the Tree (BFS Traversal)
    while !isempty(queue)
        node = popfirst!(queue)
        push!(node_list, node)
        region_to_idx[objectid(node)] = length(node_list) - 1
        append!(queue, node.children)
    end
    
    n_nodes = length(node_list)
    
    # 2. Pre-allocate Arrays
    parent_ids = Vector{Int}(undef, n_nodes)
    layer_idxs = Vector{Int}(undef, n_nodes)
    volumes    = Vector{Float64}(undef, n_nodes)
    bounded    = Vector{Bool}(undef, n_nodes)
    centroids  = Matrix{Float64}(undef, n_nodes, tree.input_dim)
    
    qlw_flat = Int[]
    qlw_offsets = Vector{Int}(undef, n_nodes + 1)
    qlw_offsets[1] = 0
    
    for (i, node) in enumerate(node_list)
        parent_ids[i] = isnothing(node.parent) ? -1 : region_to_idx[objectid(node.parent)]
        layer_idxs[i] = node.layer_number
        volumes[i]    = node.volume
        bounded[i]    = node.bounded
        centroids[i, :] = node.x
        
        append!(qlw_flat, node.qlw)
        qlw_offsets[i+1] = length(qlw_flat)
    end

    # CRITICAL FIX: Transpose centroids so Python (row-major) reads it as (N, D)
    centroids_to_save = collect(centroids')

    # 3. Dimensionality-Aware Write Helper
    function write_dataset(name, val)
        if haskey(g, name)
            delete_object(g, name)
        end
        
        dims = size(val)
        n_dims = ndims(val)
        
        if n_dims == 1
            c_size = min(dims[1], 1000)
            chunk_dims = (c_size,)
        else
            c_size = min(dims[2], 1000)
            chunk_dims = (dims[1], c_size)
        end
        
        ds = create_dataset(g, name, datatype(val), dataspace(val); 
                            chunk=chunk_dims, compress=3)
        write(ds, val)
    end
    
    # 4. Write all datasets
    write_dataset("parent_ids", parent_ids)
    write_dataset("layer_idxs", layer_idxs)
    write_dataset("volumes", volumes)
    write_dataset("bounded", bounded)
    write_dataset("centroids", centroids_to_save)
    write_dataset("qlw_flat", qlw_flat)
    write_dataset("qlw_offsets", qlw_offsets)
end

end # module