using PyCall
using Printf
using Dates
using LinearAlgebra
# Force BLAS to use 1 thread, allowing Julia threads to manage parallelism
BLAS.set_num_threads(1)

# 1. Load your local PolyhedraTree module
include("gb_julia/PolyhedraTree.jl")
using .PolyhedraTree

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
        
        start_t = time()
        println("--- Epoch $epoch ---")
        
        # New:
        tree = PolyhedraTree.Tree(state)
        PolyhedraTree.construct_tree!(tree, verbose=true)
        trees[epoch] = tree
        
        end_t = time()
        @printf("Duration: %.2f s\n", end_t - start_t)
    end
    
    tot_end = time()
    @printf("Total duration: %.2f s\n", tot_end - tot_start)
    
    return trees
end


# Ensure multi-threading is on
if Threads.nthreads() == 1
    println("Warning: Running on 1 thread. Run with 'julia --threads auto main.jl' for speed.")
end
# println(Threads.nthreads())

trees = main()


using Polyhedra
using CDDLib      # The backend solver (C-library)
using Plots
using Colors
using LinearAlgebra

# 1. Helper to create a bounded polyhedron from a Region
function get_bounded_polyhedron(region; bound=2000)
    # Get the inequalities from your struct: Dlw * x <= glw
    # We explicitly copy them to ensure we don't mutate the tree
    A, b = PolyhedraTree.get_path_inequalities(region)
    
    # Dimensions
    dim = size(A, 2)
    
    # Create Bounding Box Constraints: -bound <= x <= bound
    # This is equivalent to:  I*x <= bound  AND -I*x <= bound
    I_mat = Matrix{Float64}(I, dim, dim)
    
    A_box = vcat(I_mat, -I_mat)      # Stack identity and negative identity
    b_box = fill(bound, 2 * dim)     # Vector of [bound, bound, ...]
    
    # Combine Region constraints with Box constraints
    A_full = vcat(A, A_box)
    b_full = vcat(b, b_box)
    
    # Create the Polyhedron using CDDLib
    # H-representation: {x | Ax <= b}
    h = hrep(A_full, b_full)
    poly = polyhedron(h, CDDLib.Library())
    
    # Compute vertices (V-representation) to check if empty/valid
    # removevredundancy! triggers the computation of vertices
    try
        removevredundancy!(poly)
        if isempty(poly)
            return nothing
        end
        return poly
    catch e
        return nothing
    end
end

# 2. Main Plotting Function
function plot_epoch_layer_grid(trees; bound=10)
    epochs = sort(collect(keys(trees)))
    num_epochs = length(epochs)
    
    # Assume all trees have same depth
    num_layers = trees[epochs[1]].L
    
    # Setup the plot layout (Grid: Layers x Epochs)
    # layout = @layout [grid(num_layers, num_epochs)]
    
    # Initialize a large plot object with subplots
    p = plot(layout = (num_layers, num_epochs), 
             size = (num_epochs * 300, num_layers * 300),
             legend = false,
             framestyle = :box)

    for (col, epoch) in enumerate(epochs)
        tree = trees[epoch]
        
        for layer in 1:num_layers
            # Calculate subplot index (linear indexing in Julia Plots)
            # Row `layer`, Column `col`
            subplot_idx = (layer - 1) * num_epochs + col
            
            # Formatting (Titles and Labels)
            if layer == 1
                plot!(p[subplot_idx], title="Epoch $epoch")
            end
            if col == 1
                plot!(p[subplot_idx], ylabel="Layer $layer")
            end
            
            # Set limits and aspect ratio
            plot!(p[subplot_idx], 
                  xlims=(-bound, bound), 
                  ylims=(-bound, bound), 
                  aspect_ratio=:equal,
                  grid=false)

            # Get regions and plot them
            # Note: Explicitly calling PolyhedraTree module if needed
            regions = PolyhedraTree.get_regions_at_layer(tree, layer)
            
            for region in regions
                poly = get_bounded_polyhedron(region)
                
                if !isnothing(poly)
                    # Plots.jl has a recipe for Polyhedra. 
                    # It handles the triangulation/polygon creation automatically.
                    # Inside your plotting loop:
                    plot!(p[subplot_idx], poly, 
                        color = rand(RGB), 
                        alpha = 0.4,          # Lower alpha makes overlaps appear darker
                        linecolor = :black, 
                        linewidth = 1.5)      # Thicker lines show the partition structure better
                                    end
            end
        end
    end
    
    return p
end

# Usage:
plt = plot_epoch_layer_grid(trees, bound=2.1)
display(plt)
# savefig(plt, "grid_visualization.png")



using LinearAlgebra
using ProgressMeter
using Statistics  # <--- This fixes the UndefVarError

function check_overlaps(tree, layer_idx; method=:center, tol=1e-6)
    println("\n--- Checking Overlaps for Layer $layer_idx ---")
    
    regions = PolyhedraTree.get_regions_at_layer(tree, layer_idx)
    n = length(regions)
    println("Found $n regions. Preparing geometry...")
    
    region_data = []
    
    @showprogress for r in regions
        # Get full path constraints
        A, b = PolyhedraTree.get_path_inequalities(r)
        
        # Calculate a representative interior point
        poly = get_bounded_polyhedron(r, bound=10.0)
        
        if isnothing(poly) || isempty(poly)
            push!(region_data, nothing)
            continue
        end
        
        verts = collect(points(vrep(poly)))
        if isempty(verts)
            push!(region_data, nothing)
            continue
        end
        
        # Compute centroid (average of vertices)
        # This function now works because Statistics is loaded
        center = mean(verts)
        
        push!(region_data, (A=A, b=b, center=center, id=r.qlw))
    end
    
    # 2. Check for Overlaps
    overlap_count = 0
    overlapping_pairs = Set{Tuple{Int, Int}}()
    
    println("Testing intersections...")
    
    for i in 1:n
        data_i = region_data[i]
        isnothing(data_i) && continue
        
        center_i = data_i.center
        
        for j in 1:n
            i == j && continue 
            
            data_j = region_data[j]
            isnothing(data_j) && continue
            
            # Check: Is center_i satisfying constraints of region j?
            violation = data_j.A * center_i - data_j.b
            
            # If max violation is negative, point is INSIDE
            if maximum(violation) < -tol
                pair = minmax(i, j)
                if !(pair in overlapping_pairs)
                    push!(overlapping_pairs, pair)
                    overlap_count += 1
                    if overlap_count <= 5
                        println("  !! Overlap detected: Region $i and Region $j")
                    end
                end
            end
        end
    end
    
    if overlap_count == 0
        println("✅ No interior overlaps detected.")
    else
        println("❌ Found $overlap_count overlapping pairs!")
    end
    
    return overlap_count
end


for layer in 1:5
    check_overlaps(trees[0], layer) # Check epoch 0, layer 2
end
