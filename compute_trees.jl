using HDF5
using Printf
using LinearAlgebra

# Load your local PolyhedraTree module
include("geobin_jl/geobin.jl")
using .Geobin

function process_trees_in_h5(filename::String)
    println("Processing HDF5 file: $filename")

    # 1. SCAN: Get list of epochs to process
    # We open briefly just to get group names, then close immediately.
    group_names = h5open(filename, "r") do file
        filter(k -> startswith(k, "epoch_"), keys(file))
    end
    
    # Sort numerically (epoch_0, epoch_10, etc.)
    sort!(group_names, by = x -> parse(Int, split(x, "_")[2]))

    for g_name in group_names
        println("\n=== Processing $g_name ===")
        
        # ---------------------------------------------------------
        # PHASE 1: READ MODEL (Quick I/O)
        # ---------------------------------------------------------
        println("  - Reading weights from HDF5...")
        fake_state_dict = Dict{String, Any}()
        
        h5open(filename, "r") do file
            g = file[g_name]
            model_group = g["model"]
            
            for k in keys(model_group)
                data = read(model_group[k])
                
                # Transpose W: 
                # Python HDF5 (Out, In) -> Julia Read (In, Out) -> Transpose back to (Out, In)
                if startswith(k, "W_")
                    data = permutedims(data, (2, 1))
                    layer_num = split(k, "_")[2]
                    fake_state_dict["layer_$(layer_num)_weight"] = data
                elseif startswith(k, "b_")
                    layer_num = split(k, "_")[2]
                    fake_state_dict["layer_$(layer_num)_bias"] = data
                end
            end
        end
        # FILE IS NOW CLOSED.
        
        # ---------------------------------------------------------
        # PHASE 2: COMPUTE TREE (Slow, CPU Intensive)
        # ---------------------------------------------------------
        # This happens purely in memory.
        
        tree = Tree(fake_state_dict)
        
        t_start = time()
        construct_tree!(tree, verbose=false)
        duration = time() - t_start
        
        leaves = get_regions_at_layer(tree, tree.L)
        n_leaves = length(leaves)
        @printf("  - Tree constructed in %.2fs. Leaves: %d\n", duration, n_leaves)
        
        # ---------------------------------------------------------
        # PHASE 3: PREPARE DATA ARRAYS (In Memory)
        # ---------------------------------------------------------
        # Flatten the tree structures into arrays before opening the file
        println("  - Linearizing tree data...")
        
        node_list = Region[]
        queue = [tree.root]
        region_to_idx = Dict{UInt64, Int}()
        
        while !isempty(queue)
            node = popfirst!(queue)
            push!(node_list, node)
            region_to_idx[objectid(node)] = length(node_list) - 1 # 0-based index
            append!(queue, node.children)
        end
        
        n_nodes = length(node_list)
        
        # Allocate arrays
        parent_ids = Vector{Int}(undef, n_nodes)
        layer_idxs = Vector{Int}(undef, n_nodes)
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
            volumes[i]    = node.volume
            bounded[i]    = node.bounded
            centroids[i, :] = node.x
            
            # Activations
            append!(qlw_flat, node.qlw)
            qlw_offsets[i+1] = length(qlw_flat)
        end

        # ---------------------------------------------------------
        # PHASE 4: WRITE DATA (Quick I/O)
        # ---------------------------------------------------------
        println("  - Writing tree topology to HDF5...")
        
        h5open(filename, "r+") do file
            g = file[g_name]
            
            # Helper to overwrite if dataset exists
            function write_dataset(name, val)
                if haskey(g, name)
                    delete_object(g, name)
                end
                g[name] = val
            end
            
            write_dataset("parent_ids", parent_ids)
            write_dataset("layer_idxs", layer_idxs)
            write_dataset("volumes", volumes)
            write_dataset("bounded", bounded)
            write_dataset("centroids", centroids)
            write_dataset("qlw_flat", qlw_flat)
            write_dataset("qlw_offsets", qlw_offsets)
        end
        # FILE IS NOW CLOSED.
        
        # Force garbage collection to free memory before next epoch
        tree = nothing
        fake_state_dict = nothing
        node_list = nothing
        GC.gc()
    end
    
    println("\nAll trees processed and saved successfully.")
end

# Run
process_trees_in_h5("test_experiment.h5")