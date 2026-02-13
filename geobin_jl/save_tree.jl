module SaveTree

using HDF5
using ..Regions

export save_trees_to_hdf5

"""
    save_trees_to_hdf5(filename::String, trees::Dict{Int, Any})

Saves a dictionary of PolyhedraTrees to an HDF5 file using a linearized 
schema for maximum efficiency and easy Python reconstruction.
"""
function save_trees_to_hdf5(filename::String, trees::Dict{Int, Any})
    h5open(filename, "w") do file
        for (epoch, tree) in trees
            # Create a group for this epoch
            g = create_group(file, "epoch_$epoch")
            
            # 1. Save Global Model Parameters (Weights/Biases)
            # We save these so Python can reconstruct geometry if needed
            g_model = create_group(g, "model")
            for l in 1:length(tree.weights)
                g_model["W_$(l)"] = tree.weights[l]
                g_model["b_$(l)"] = tree.biases[l]
            end
            g["input_dim"] = tree.input_dim
            g["depth"] = tree.L

            # 2. Linearize the Tree (BFS Traversal)
            # We map every Region object to a unique integer ID (0 to N-1)
            # Root is always 0
            
            node_list = Region[]
            queue = [tree.root]
            
            # Map object ID to Index
            region_to_idx = Dict{UInt64, Int}()
            
            # BFS collection
            while !isempty(queue)
                node = popfirst!(queue)
                push!(node_list, node)
                region_to_idx[objectid(node)] = length(node_list) - 1 # 0-based index
                append!(queue, node.children)
            end
            
            n_nodes = length(node_list)
            println("  - Saving Epoch $epoch: $n_nodes nodes...")

            # 3. Pre-allocate Arrays for HDF5
            # Topology
            parent_ids = Vector{Int}(undef, n_nodes)
            layer_idxs = Vector{Int}(undef, n_nodes)
            is_leaf    = Vector{Bool}(undef, n_nodes)
            
            # Data
            volumes    = Vector{Float64}(undef, n_nodes)
            bounded    = Vector{Bool}(undef, n_nodes)
            centroids  = Matrix{Float64}(undef, n_nodes, tree.input_dim)
            
            # Ragged Array for Activations (qlw)
            # Store as one giant 1D array + an offset array
            qlw_flat = Int[]
            qlw_offsets = Vector{Int}(undef, n_nodes + 1)
            qlw_offsets[1] = 0 # 0-based offset for Python compatibility

            for (i, node) in enumerate(node_list)
                # i is 1-based, but our IDs are 0-based
                idx = i - 1
                
                # Topology
                if isnothing(node.parent)
                    parent_ids[i] = -1
                else
                    parent_ids[i] = region_to_idx[objectid(node.parent)]
                end
                
                layer_idxs[i] = node.layer_number
                is_leaf[i]    = isempty(node.children)
                
                # Metrics
                volumes[i]    = node.volume
                bounded[i]    = node.bounded
                centroids[i, :] = node.x # The feasible point inside
                
                # Flatten Activations
                append!(qlw_flat, node.qlw)
                qlw_offsets[i+1] = length(qlw_flat)
            end

            # 4. Write Arrays to HDF5
            g["parent_ids"]  = parent_ids
            g["layer_idxs"]  = layer_idxs
            g["is_leaf"]     = is_leaf
            g["volumes"]     = volumes
            g["bounded"]     = bounded
            g["centroids"]   = centroids # shape (N, Dim)
            
            # Write Ragged Activations
            g["qlw_flat"]    = qlw_flat
            g["qlw_offsets"] = qlw_offsets
        end
    end
    println("Successfully saved trees to $filename")
end

# Usage example (Run this after your main loop):
# save_trees_to_hdf5("experiment_trees.h5", trees)

end #module