using HDF5
using Printf
using YAML
using ArgParse
using Base.Threads
using LinearAlgebra
using Dates # For timing

# 1. Force BLAS to 1 thread to avoid contention with Julia threads
# BLAS.set_num_threads(1)

include("geobin_jl/geobin.jl")
using .Geobin

function main()
    # --- ARGS PARSING ---
    s = ArgParseSettings()
    @add_arg_table s begin
        "config"
            help = "Path to the YAML configuration file"
            required = true
            arg_type = String
        "--overwrite"
            help = "Force re-computation"
            action = :store_true
    end
    parsed_args = parse_args(ARGS, s)
    config_path = parsed_args["config"]
    overwrite = parsed_args["overwrite"]

    # --- SETUP PATHS ---
    config = YAML.load_file(config_path)
    output_dir = get(config, "output_dir", ".")
    exp_name = get(config, "experiment_name", "experiment")
    
    # The ultimate destination on Network Storage
    network_h5_path = joinpath(output_dir, exp_name, "$(exp_name).h5")
    
    # The temporary location on Local Storage (Node SSD)
    # uses /tmp or the directory defined by TMPDIR env var
    temp_h5_path = joinpath(tempdir(), "$(exp_name)_$(getpid()).h5")

    println("="^60)
    println("🚀 GEOBIN CLUSTER RUNNER")
    println("="^60)
    println("• Network Path: $network_h5_path")
    println("• Local Scratch: $temp_h5_path")
    println("• Threads:      $(Threads.nthreads())")
    println("-"^60)

    # --- STAGING: COPY TO LOCAL ---
    if isfile(network_h5_path)
        println("📦 Staging: Copying file to local scratch...")
        t_copy = @elapsed cp(network_h5_path, temp_h5_path; force=true)
        println("   └─ Done in $(round(t_copy, digits=2))s")
    else
        # If file doesn't exist (e.g. Python didn't run), create new locally
        println("⚠️ Network file not found. Creating new local file.")
        h5open(temp_h5_path, "w") do file
            create_group(file, "epochs")
        end
    end

    try
        # --- PROCESSING (ON LOCAL FILE) ---
        process_local_file(temp_h5_path, overwrite)

        # --- COMMIT: COPY BACK TO NETWORK ---
        println("\n💾 Commit: Copying results back to network storage...")
        t_commit = @elapsed cp(temp_h5_path, network_h5_path; force=true)
        println("   └─ Done in $(round(t_commit, digits=2))s")
        
    catch e
        println("\n❌ ERROR during processing:")
        showerror(stdout, e, catch_backtrace())
        exit(1)
        
    finally
        # --- CLEANUP ---
        if isfile(temp_h5_path)
            println("🧹 Cleanup: Removing local scratch file.")
            rm(temp_h5_path)
        end
    end
    
    println("\n✅ Execution Complete.")
end

function process_local_file(h5_path::String, overwrite::Bool)
    # Scan for epochs inside the LOCAL file
    group_names = h5open(h5_path, "r") do file
        if !haskey(file, "epochs"); return String[]; end
        filter(k -> startswith(k, "epoch_"), keys(file["epochs"]))
    end
    sort!(group_names, by = x -> parse(Int, split(x, "_")[2]))

    println("\nProcessing $(length(group_names)) epochs...")

    for g_name in group_names
        # 1. Check if done
        already_done = false
        h5open(h5_path, "r") do file
            if haskey(file["epochs"][g_name], "centroids")
                already_done = true
            end
        end
        if already_done && !overwrite
            println("  - $g_name: Skipped (Exists)")
            continue
        end

        println("\n  ▶ $g_name")
        
        # 2. Read Weights (Fast Local Read)
        fake_state_dict = Dict{String, Any}()
        h5open(h5_path, "r") do file
            g = file["epochs"][g_name]
            for k in keys(g)
                if occursin("weight", k) || occursin("bias", k)
                    data = read(g[k])
                    if occursin("weight", k) && ndims(data) == 2
                        data = permutedims(data, (2, 1))
                    end
                    fake_state_dict[k] = data
                end
            end
        end

        # 3. Construct Tree
        t_start = time()
        tree = Tree(fake_state_dict)
        construct_tree!(tree)
        dt = time() - t_start
        
        n_nodes = length(Geobin.get_children(tree.root)) # Approx node count check
        println("    └─ Tree Built: $(round(dt, digits=2))s | Nodes: ~$n_nodes")

        # 4. Save (Fast Local Write)
        # Using the helper from save_tree.jl we defined earlier
        save_single_tree_to_hdf5(h5_path, tree, g_name)
        println("    └─ Saved to disk.")

        # 5. Memory Cleanup
        tree = nothing
        fake_state_dict = nothing
        GC.gc() 
    end
end

main()