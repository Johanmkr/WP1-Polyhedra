module SaveTree

using HDF5
using ..Regions
using ..Trees

export save_single_tree_to_hdf5

"""
    save_single_tree_to_hdf5(filename::String, tree::Tree, epoch_name::String)

Appends a SINGLE tree to the HDF5 file. 
Designed to be called immediately after tree construction to allow memory cleanup.
"""
function save_single_tree_to_hdf5(filename::String, tree::Tree, epoch_name::String)
    # Open in read/write mode ("r+") to append to existing file
    h5open(filename, "r+") do file
        # Check if group exists and delete if necessary (overwrite mode)
        if haskey(file, "epochs/$epoch_name")
            delete_object(file, "epochs/$epoch_name")
        end
        
        # Create group
        # Ensure 'epochs' group exists
        if !haskey(file, "epochs")
            create_group(file, "epochs")
        end
        
        g = create_group(file["epochs"], epoch_name)
        
        # 1. Save Global Model Parameters
        g_model = create_group(g, "model")
        for l in 1:length(tree.weights)
            g_model["W_$(l)"] = tree.weights[l]
            g_model["b_$(l)"] = tree.biases[l]
        end
        g["input_dim"] = tree.input_dim
        g["depth"] = tree.L

        # 2. Linearize Tree
        node_list = Region[]
        queue = [tree.root]
        region_to_idx = Dict{UInt64, Int}()
        
        while !isempty(queue)
            node = popfirst!(queue)
            push!(node_list, node)
            region_to_idx[objectid(node)] = length(node_list) - 1
            append!(queue, node.children)
        end
        
        n_nodes = length(node_list)
        
        # 3. Prepare Arrays
        parent_ids = Vector{Int}(undef, n_nodes)
        layer_idxs = Vector{Int}(undef, n_nodes)
        is_leaf    = Vector{Bool}(undef, n_nodes)
        volumes    = Vector{Float64}(undef, n_nodes)
        bounded    = Vector{Bool}(undef, n_nodes)
        centroids  = Matrix{Float64}(undef, n_nodes, tree.input_dim)
        
        qlw_flat = Int[]
        qlw_offsets = Vector{Int}(undef, n_nodes + 1)
        qlw_offsets[1] = 0

        for (i, node) in enumerate(node_list)
            # Topology
            parent_ids[i] = isnothing(node.parent) ? -1 : region_to_idx[objectid(node.parent)]
            layer_idxs[i] = node.layer_number
            is_leaf[i]    = isempty(node.children)
            
            # Metrics
            volumes[i]    = node.volume
            bounded[i]    = node.bounded
            centroids[i, :] = node.x
            
            # Activations
            append!(qlw_flat, node.qlw)
            qlw_offsets[i+1] = length(qlw_flat)
        end

        # 4. Write Data with Compression
        # Helper to write compressed datasets
        function write_ds(name, data)
            dims = size(data)
            # Chunking strategy: try to chunk by ~1000 items
            if ndims(data) == 1
                chunk = (min(dims[1], 1024),)
            else
                chunk = (min(dims[1], 1024), dims[2]) # Chunk rows
            end
            ds = create_dataset(g, name, datatype(data), dataspace(data); chunk=chunk, compress=3)
            write(ds, data)
        end

        write_ds("parent_ids", parent_ids)
        write_ds("layer_idxs", layer_idxs)
        write_ds("is_leaf", is_leaf)
        write_ds("volumes", volumes)
        write_ds("bounded", bounded)
        write_ds("centroids", centroids)
        write_ds("qlw_flat", qlw_flat)
        write_ds("qlw_offsets", qlw_offsets)
    end
end

end #module