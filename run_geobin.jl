using HDF5
using Printf
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

    mode_str = overwrite ? "ON" : "OFF"
    println("Processing HDF5 file: $filename")
    println("Overwrite mode: $mode_str")
    println("Running with $(Threads.nthreads()) threads.")

    # We open the file ONCE in read-write mode to process everything
    h5open(filename, "r+") do file
        if !haskey(file, "epochs")
            println("Error: 'epochs' group not found in HDF5 file.")
            return
        end
        
        # 1. SCAN: Get list of epochs
        group_names = filter(k -> startswith(k, "epoch_"), keys(file["epochs"]))
        if isempty(group_names)
            println("No epochs found to process.")
            return
        end
        
        # Sort numerically (epoch_0, epoch_1, etc.)
        sort!(group_names, by = x -> parse(Int, split(x, "_")[2]))

        for g_name in group_names
            println("\n=== Processing $g_name ===")
            g = file["epochs"][g_name]
            
            # ---------------------------------------------------------
            # PHASE 0: CHECK EXISTENCE
            # ---------------------------------------------------------
            if haskey(g, "parent_ids")
                if !overwrite
                    println("  - Tree data found. Skipping. (Pass --overwrite to recompute)")
                    continue
                else
                    println("  - Tree data found. Overwriting...")
                end
            end

            # ---------------------------------------------------------
            # PHASE 1: READ MODEL
            # ---------------------------------------------------------
            state_dict = read_state_dict_from_h5(g)
            if isempty(state_dict)
                println("  Warning: No model weights found in $g_name. Skipping.")
                continue
            end
            
            # ---------------------------------------------------------
            # PHASE 2: COMPUTE TREE
            # ---------------------------------------------------------
            tree = Tree(state_dict)
            
            t_start = time()
            construct_tree!(tree, verbose=false)
            duration = time() - t_start
            
            leaves = get_regions_at_layer(tree, tree.L)
            @printf("  - Tree constructed in %.2fs. Leaves: %d\n", duration, length(leaves))
            
            # ---------------------------------------------------------
            # PHASE 3: WRITE DATA
            # ---------------------------------------------------------
            save_tree_to_h5(g, tree)
            
            # Garbage collection
            tree = nothing
            state_dict = nothing
            GC.gc()
        end
        println("\nAll trees processed.")
    end
end

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "h5_file"
            help = "Path to the HDF5 file containing the experiment data"
            required = true
            arg_type = String
        "--overwrite"
            help = "Force re-computation even if tree data exists"
            action = :store_true
    end
    
    parsed_args = parse_args(ARGS, s)
    h5_filename = parsed_args["h5_file"]
    should_overwrite = parsed_args["overwrite"]

    process_trees_in_h5(h5_filename; overwrite=should_overwrite)
end

main()