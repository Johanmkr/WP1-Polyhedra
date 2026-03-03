module SaveTree

using HDF5
using ..Regions
using ..Trees

export read_state_dict_from_h5, save_tree_to_h5

function read_state_dict_from_h5(ep_group::HDF5.Group)
    state_dict = Dict{String, Any}()
    for k in keys(ep_group)
        if occursin("weight", k)
            data = read(ep_group[k])
            if ndims(data) == 2
                data = permutedims(data, (2, 1))
            end
            state_dict[k] = data
        elseif occursin("bias", k)
            state_dict[k] = read(ep_group[k])
        end
    end
    return state_dict
end

"""
    save_tree_to_h5(g::HDF5.Group, tree::Tree)
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
    parent_ids = Vector{Int32}(undef, n_nodes)
    layer_idxs = Vector{Int8}(undef, n_nodes)
    volumes_ex    = Vector{Float64}(undef, n_nodes)
    volumes_es    = Vector{Float64}(undef, n_nodes)
    bounded    = Vector{Bool}(undef, n_nodes)
    centroids  = Matrix{Float64}(undef, n_nodes, tree.input_dim)
    
    # Flat lists for dynamic size arrays
    qlw_flat = Int8[]
    qlw_offsets = Vector{Int64}(undef, n_nodes + 1)
    qlw_offsets[1] = 0
    
    active_flat = Int32[]
    active_offsets = Vector{Int64}(undef, n_nodes + 1)
    active_offsets[1] = 0
    
    for (i, node) in enumerate(node_list)
        parent_ids[i] = isnothing(node.parent) ? -1 : region_to_idx[objectid(node.parent)]
        layer_idxs[i] = node.layer_number
        volumes_ex[i]    = node.volume_ex
        volumes_es[i]    = node.volume_es
        bounded[i]    = node.bounded
        centroids[i, :] = node.x
        
        # Convert BitVector to Int8 for HDF5 storage
        append!(qlw_flat, Int8.(node.qlw))
        qlw_offsets[i+1] = length(qlw_flat)
        
        append!(active_flat, node.active_indices)
        active_offsets[i+1] = length(active_flat)
    end

    centroids_to_save = collect(centroids')

    # 3. Dimensionality-Aware Write Helper
    function write_dataset(name, val)
        if haskey(g, name)
            delete_object(g, name)
        end
        # Use chunking and compression
        chunk_dims = ndims(val) == 1 ? (min(length(val), 10000),) : (size(val, 1), min(size(val, 2), 10000))
        # Handle empty arrays safely
        if isempty(val)
            chunk_dims = ndims(val) == 1 ? (1,) : (size(val, 1), 1)
        end
        
        ds = create_dataset(g, name, datatype(val), dataspace(val); chunk=chunk_dims, compress=3)
        write(ds, val)
    end
    
    write_dataset("parent_ids", parent_ids)
    write_dataset("layer_idxs", layer_idxs)
    write_dataset("volumes_ex", volumes_ex)
    write_dataset("volumes_es", volumes_es)
    write_dataset("bounded", bounded)
    write_dataset("centroids", centroids_to_save)
    write_dataset("qlw_flat", qlw_flat)
    write_dataset("qlw_offsets", qlw_offsets)
    write_dataset("active_flat", active_flat)
    write_dataset("active_offsets", active_offsets)
end

end # module