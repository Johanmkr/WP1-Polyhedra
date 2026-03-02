# using Pkg
# Pkg.activate(".") 

using HDF5
using YAML
using ArgParse
using LinearAlgebra
using Printf

# Include project modules
include("geobin_jl/geobin.jl")
using .Geobin

# Explicitly import add_child! since it is not exported by Geobin
import .Geobin.Regions: add_child!

"""
    reconstruct_tree_from_h5(h5_path, epoch_name)
    
Loads weights and region topology from HDF5 and returns a fully hydrated Tree.
"""
function reconstruct_tree_from_h5(h5_path, epoch_name)
    println("  Loading $epoch_name from $h5_path...")
    
    fake_state_dict = Dict{String, Any}()
    
    # 1. Load Weights
    h5open(h5_path, "r") do file
        if !haskey(file, "epochs/$epoch_name")
            error("Epoch $epoch_name not found in $h5_path")
        end
        
        g = file["epochs/$epoch_name"]
        
        src = haskey(g, "model") ? g["model"] : g
        
        for k in keys(src)
            if occursin("W_", k) || occursin("weight", k) || occursin("b_", k) || occursin("bias", k)
                data = read(src[k])
                fake_state_dict[k] = data
            end
        end
        
        # 2. Load Region Data
        global parent_ids = read(g["parent_ids"])
        global layer_idxs = read(g["layer_idxs"])
        global centroids  = read(g["centroids"])
        global qlw_flat   = read(g["qlw_flat"])
        global qlw_offsets = read(g["qlw_offsets"])
        global volumes    = read(g["volumes"])
        global bounded    = read(g["bounded"])
    end
    
    # 3. Create Skeleton Tree
    tree = Tree(fake_state_dict)
    
    # 4. Rebuild Nodes and Linkage
    nodes = Vector{Region}(undef, length(parent_ids))
    
    # First pass: Create Nodes
    for i in 1:length(parent_ids)
        start_idx = qlw_offsets[i] + 1
        end_idx   = qlw_offsets[i+1]
        act = qlw_flat[start_idx:end_idx]
        
        r = Region(act)
        r.layer_number = layer_idxs[i]
        r.volume = volumes[i]
        r.bounded = bounded[i] > 0
        r.x = centroids[i, :] 
        
        nodes[i] = r
    end
    
    # Second pass: Linkage
    for i in 1:length(parent_ids)
        pid = parent_ids[i]
        child = nodes[i]
        
        if pid == -1
            tree.root = child
            
            # --- ROOT INITIALIZATION ---
            child.Alw = Matrix{Float64}(I, tree.input_dim, tree.input_dim)
            child.clw = zeros(tree.input_dim)
            child.Dlw = Matrix{Float64}(undef, 0, tree.input_dim)
            child.glw = Float64[]
            
            # FIX: Initialize active constraints for Root (Empty but defined)
            # This prevents UndefRefError during traversal
            child.Dlw_active = child.Dlw
            child.glw_active = child.glw
        else
            parent = nodes[pid + 1] 
            add_child!(parent, child)
        end
    end
    
    # 5. Hydrate Geometry
    hydrate_geometry!(tree)
    
    return tree
end

function hydrate_geometry!(tree::Tree)
    queue = [tree.root]
    
    while !isempty(queue)
        parent = popfirst!(queue)
        
        if isempty(parent.children); continue; end
        if parent.layer_number >= tree.L; continue; end
        
        W = tree.weights[parent.layer_number + 1]
        b = tree.biases[parent.layer_number + 1]
        
        W_hat = W * parent.Alw
        b_hat = W * parent.clw + b
        
        for child in parent.children
            q = child.qlw
            
            s_vec = -2.0 .* q .+ 1.0
            
            child.Dlw = s_vec .* W_hat
            child.glw = -(s_vec .* b_hat)
            
            child.Alw = q .* W_hat
            child.clw = q .* b_hat
            
            child.Dlw_active = child.Dlw
            child.glw_active = child.glw
            
            push!(queue, child)
        end
    end
end

function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "config"
            help = "Path to the YAML configuration file"
            required = true
            arg_type = String
        "--bound"
            help = "Bounding box size for Monte Carlo"
            default = 10.0
            arg_type = Float64
    end
    parsed_args = parse_args(ARGS, s)
    config_path = parsed_args["config"]
    bound = parsed_args["bound"]

    config = YAML.load_file(config_path)
    output_dir = get(config, "output_dir", ".")
    exp_name = get(config, "experiment_name", "experiment")
    h5_filename = joinpath(output_dir, exp_name, "$(exp_name).h5")
    
    if !isfile(h5_filename)
        println("❌ File not found: $h5_filename")
        exit(1)
    end

    println("🔍 Verifying: $h5_filename")
    println("   Threads:  $(Threads.nthreads())")
    
    epoch_names = h5open(h5_filename, "r") do file
        filter(k -> startswith(k, "epoch_"), keys(file["epochs"]))
    end
    sort!(epoch_names, by = x -> parse(Int, split(x, "_")[2]))
    
    for ename in epoch_names
        println("\n" * "-"^60)
        println("PROCESSING $ename")
        println("-"^60)
        
        try
            tree = reconstruct_tree_from_h5(h5_filename, ename)
            verify_tree_properties(tree; bound=bound)
        catch e
            println("❌ Error verifying $ename:")
            showerror(stdout, e, catch_backtrace())
        end
        
        GC.gc()
    end
end

main()