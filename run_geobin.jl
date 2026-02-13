using HDF5
using Printf
using LinearAlgebra
using YAML
using ArgParse

# Load your local PolyhedraTree module
include("geobin_jl/geobin.jl")
using .Geobin

"""
    process_trees_in_h5(filename::String; overwrite::Bool=false)

Scans the HDF5 file for 'epoch_X' groups. 
If a tree has already been computed (checked via existence of 'parent_ids'), 
it skips computation unless `overwrite` is set to true.
"""
function process_trees_in_h5(filename::String; overwrite::Bool=false)
    if !isfile(filename)
        println("❌ Error: HDF5 file not found at: $filename")
        exit(1)
    end

    # Fix: Clean up interpolation to avoid Juxtaposition Errors
    mode_str = overwrite ? "ON" : "OFF"
    println("Processing HDF5 file: $filename")
    println("Overwrite mode: $mode_str")
    println("Running with $(Threads.nthreads()) threads.")

    # 1. SCAN: Get list of epochs
    group_names = h5open(filename, "r") do file
        if !haskey(file, "epochs")
            println("Error: 'epochs' group not found in HDF5 file.")
            return String[]
        end
        filter(k -> startswith(k, "epoch_"), keys(file["epochs"]))
    end
    
    if isempty(group_names)
        println("No epochs found to process.")
        return
    end
    
    # Sort numerically (epoch_0, epoch_1, etc.)
    sort!(group_names, by = x -> parse(Int, split(x, "_")[2]))

    for g_name in group_names
        println("\n=== Processing $g_name ===")
        
        # ---------------------------------------------------------
        # PHASE 0: CHECK EXISTENCE
        # ---------------------------------------------------------
        already_exists = false
        h5open(filename, "r") do file
            g = file["epochs"][g_name]
            if haskey(g, "parent_ids")
                already_exists = true
            end
        end

        if already_exists && !overwrite
            println("  - Tree data found. Skipping. (Pass --overwrite to recompute)")
            continue
        elseif already_exists && overwrite
            println("  - Tree data found. Overwriting...")
        end

        # ---------------------------------------------------------
        # PHASE 1: READ MODEL
        # ---------------------------------------------------------
        fake_state_dict = Dict{String, Any}()
        
        h5open(filename, "r") do file
            ep_group = file["epochs"][g_name]
            
            for k in keys(ep_group)
                if occursin("weight", k)
                    data = read(ep_group[k])
                    # Transpose PyTorch weights (Out, In) -> Julia (In, Out) -> Transpose back
                    if ndims(data) == 2
                        data = permutedims(data, (2, 1))
                    end
                    fake_state_dict[k] = data
                elseif occursin("bias", k)
                    data = read(ep_group[k])
                    fake_state_dict[k] = data
                end
            end
        end
        
        if isempty(fake_state_dict)
            println("  Warning: No model weights found in $g_name. Skipping.")
            continue
        end
        
        # ---------------------------------------------------------
        # PHASE 2: COMPUTE TREE
        # ---------------------------------------------------------
        tree = Tree(fake_state_dict)
        
        t_start = time()
        construct_tree!(tree, verbose=false)
        duration = time() - t_start
        
        leaves = get_regions_at_layer(tree, tree.L)
        @printf("  - Tree constructed in %.2fs. Leaves: %d\n", duration, length(leaves))
        
        # ---------------------------------------------------------
        # PHASE 3: LINEARIZE DATA
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # PHASE 4: WRITE DATA (with Dimensionality-Aware Compression)
        # ---------------------------------------------------------
        h5open(filename, "r+") do file
            g = file["epochs"][g_name]
            
            function write_dataset(name, val)
                # 1. Remove old data if it exists
                if haskey(g, name)
                    delete_object(g, name)
                end
                
                # 2. Determine Chunking based on Dimensions
                # Chunk size should match the dimensionality of 'val'
                dims = size(val)
                n_dims = ndims(val)
                
                # Create a chunk tuple. We cap chunks at 1000 or the actual size.
                # For 1D: (chunk_size,)
                # For 2D: (chunk_size, dims[2])
                if n_dims == 1
                    c_size = min(dims[1], 1000)
                    chunk_dims = (c_size,)
                else
                    # For centroids (D, N), we chunk along the 'Nodes' dimension (dim 2)
                    c_size = min(dims[2], 1000)
                    chunk_dims = (dims[1], c_size)
                end
                
                # 3. Create and write compressed dataset
                # We use 'compress=3' for a good balance of speed vs size
                ds = create_dataset(g, name, datatype(val), dataspace(val); 
                                    chunk=chunk_dims, compress=3)
                write(ds, val)
            end
            
            # Write all datasets using the improved function
            write_dataset("parent_ids", parent_ids)
            write_dataset("layer_idxs", layer_idxs)
            write_dataset("volumes", volumes)
            write_dataset("bounded", bounded)
            write_dataset("centroids", centroids_to_save) # This is 2D
            write_dataset("qlw_flat", qlw_flat)
            write_dataset("qlw_offsets", qlw_offsets)
        end
        
        # Garbage collection
        tree = nothing
        fake_state_dict = nothing
        node_list = nothing
        GC.gc()
    end
    
    println("\nAll trees processed.")
end

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "config"
            help = "Path to the YAML configuration file"
            required = true
            arg_type = String
        "--overwrite"
            help = "Force re-computation even if tree data exists"
            action = :store_true
    end
    
    parsed_args = parse_args(ARGS, s)
    config_path = parsed_args["config"]
    should_overwrite = parsed_args["overwrite"]

    if !isfile(config_path)
        println("❌ Error: Config file not found at $config_path")
        exit(1)
    end

    # 1. Load YAML
    config = YAML.load_file(config_path)

    # 2. Reconstruct HDF5 path
    output_dir = get(config, "output_dir", ".")
    exp_name = get(config, "experiment_name", "experiment")
    
    h5_filename = joinpath(output_dir, exp_name, "$(exp_name).h5")

    println("Loaded Config: $config_path")
    println("Target HDF5:   $h5_filename")

    # 3. Run Processing
    process_trees_in_h5(h5_filename; overwrite=should_overwrite)
end

main()