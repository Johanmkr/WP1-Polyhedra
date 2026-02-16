using HDF5
using Printf
using YAML
using ArgParse
using Base.Threads
using LinearAlgebra
using Dates

# 1. Force BLAS to 1 thread
BLAS.set_num_threads(1)

include("geobin_jl/geobin.jl")
using .Geobin

function main()
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

    config = YAML.load_file(config_path)
    output_dir = get(config, "output_dir", ".")
    exp_name = get(config, "experiment_name", "experiment")
    
    network_h5_path = joinpath(output_dir, exp_name, "$(exp_name).h5")
    temp_h5_path = joinpath(tempdir(), "$(exp_name)_$(getpid()).h5")

    println("="^60)
    println("🚀 GEOBIN CLUSTER RUNNER")
    println("="^60)
    println("• Threads: $(Threads.nthreads())")
    println("-"^60)

    if isfile(network_h5_path)
        println("📦 Staging to local scratch...")
        cp(network_h5_path, temp_h5_path; force=true)
    else
        h5open(temp_h5_path, "w") do file
            create_group(file, "epochs")
        end
    end

    try
        process_local_file(temp_h5_path, overwrite)
        println("\n💾 Committing results...")
        cp(temp_h5_path, network_h5_path; force=true)
    catch e
        println("\n❌ ERROR:")
        showerror(stdout, e, catch_backtrace())
        exit(1)
    finally
        isfile(temp_h5_path) && rm(temp_h5_path)
    end
    
    println("\n✅ Execution Complete.")
end

function process_local_file(h5_path::String, overwrite::Bool)
    group_names = h5open(h5_path, "r") do file
        if !haskey(file, "epochs"); return String[]; end
        filter(k -> startswith(k, "epoch_"), keys(file["epochs"]))
    end
    sort!(group_names, by = x -> parse(Int, split(x, "_")[2]))

    println("\nProcessing $(length(group_names)) epochs...")

    for g_name in group_names
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

        # Read Weights
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

        # Construct
        t_start = time()
        tree = Tree(fake_state_dict)
        construct_tree!(tree)
        dt = time() - t_start
        
        # --- LOGGING UPDATES ---
        # Count total nodes and leaves
        n_nodes = length(Geobin.get_children(tree.root))
        # Leaves are regions at the final layer (L)
        leaves = Geobin.get_regions_at_layer(tree, tree.L)
        n_leaves = length(leaves)
        
        println("  ▶ $g_name | Time: $(round(dt, digits=2))s | Nodes: $n_nodes | Leaves: $n_leaves")

        save_single_tree_to_hdf5(h5_path, tree, g_name)
        
        tree = nothing
        fake_state_dict = nothing
        GC.gc() 
    end
end

main()