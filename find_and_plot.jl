using PyCall
using Printf
using Dates
using LinearAlgebra
# Force BLAS to use 1 thread, allowing Julia threads to manage parallelism
BLAS.set_num_threads(1)

# 1. Load your local PolyhedraTree module
include("geobin_jl/geobin.jl")
using .Geobin

# 2. Setup Python Environment
# Add current directory to Python path so we can find 'src_experiment'
sys = pyimport("sys")
pushfirst!(PyVector(sys."path"), ".")

# Import python libraries
torch = pyimport("torch")
src_exp = pyimport("src_experiment") # Equivalent to: import src_experiment

# Define helper to convert PyTorch state dict to Julia Dict
function convert_torch_state(py_state)
    jl_dict = Dict{String, Any}()
    for (k, v) in py_state
        # Convert Tensor -> Numpy -> Julia Array
        if pyisinstance(v, torch.Tensor)
            jl_dict[String(k)] = v.detach().cpu().numpy()
        else
            jl_dict[String(k)] = v
        end
    end
    return jl_dict
end

function main()
    # 1. Get the path object from Python
    base_path_py = src_exp.get_test_data().absolute()
    
    # --- FIX HERE ---
    # use .as_posix() to get a clean string explicitly
    base_path = base_path_py.as_posix()
    
    println("Current Working Directory: $(pwd())")
    println("Base Data Path resolved to: $base_path")

    epochs = [0, 10, 20, 30, 40]
    trees = Dict{Int, Any}()
    
    tot_start = time()
    
    for epoch in epochs
        # Construct path safely using Julia's joinpath on the clean string
        state_dict_path = joinpath(base_path, "state_dicts", "epoch$(epoch).pth")
        
        if !isfile(state_dict_path)
            println("\n!!! ERROR: File not found: $state_dict_path")
            println("Please check if the folder structure matches this path.")
            continue 
        end
        
        println("Loading: $state_dict_path")
        
        # Load using torch
        raw_state = torch.load(state_dict_path, map_location="cpu")
        state = convert_torch_state(raw_state)
        
        # Force Garbage Collection to clean up PyCall objects
        GC.gc() 

        start_t = time()
        println("--- Epoch $epoch ---")
        
        # New:
        tree = Tree(state)
        construct_tree!(tree, verbose=true)
        trees[epoch] = tree
        
        end_t = time()
        @printf("Duration: %.2f s\n", end_t - start_t)

        leaves = get_regions_at_layer(tree, tree.L)
        println("  Found $(length(leaves)) final regions.")
    end
    
    tot_end = time()
    @printf("Total duration: %.2f s\n", tot_end - tot_start)
    
    return trees
end

println("Running on $(Threads.nthreads()) threads")

trees = main()


# print tree info:
for i in eachindex(trees)
    print_tree_summary(trees[i])
    # verify_tree_properties(trees[i])
end

save_trees_to_hdf5("test_experiment.h5", trees)

