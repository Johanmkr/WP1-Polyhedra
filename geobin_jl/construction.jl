module Construction

using ..Regions
using ..Trees
using ..Geometry
using LinearAlgebra
using Statistics
using ProgressMeter

export construct_tree!

function construct_tree!(tree::Tree; verbose::Bool=false)
    current_layer_nodes = [tree.root]

    for i in 1:tree.L
        Wl = tree.weights[i]
        bl = tree.biases[i]
        layer = i

        next_layer_nodes = Region[]
        nodes_lock = ReentrantLock()

        if verbose
            println("Layer $layer: Processing $(length(current_layer_nodes)) regions...")
        end

        # Parallel loop
        # Threads.@threads for parent in current_layer_nodes
        for parent in current_layer_nodes

            new_nodes_info = find_next_layer_region_info(
                parent.Dlw_active, parent.glw_active,
                parent.Alw, parent.clw,
                Wl, bl, layer
            )

            local_children = Region[]
            for (act, info) in new_nodes_info
                child = Region(act)
                child.q_tilde = info["q_tilde"]
                child.bounded = info["bounded"]
                child.Dlw = info["Dlw"]
                child.glw = info["glw"]
                child.Dlw_active = info["Dlw_active"]
                child.glw_active = info["glw_active"]
                child.volume = info["volume"]
                child.Alw = info["Alw"]
                child.clw = info["clw"]
                child.layer_number = layer

                add_child!(parent, child)
                push!(local_children, child)
            end

            lock(nodes_lock) do
                append!(next_layer_nodes, local_children)
            end
        end

        current_layer_nodes = next_layer_nodes
        if isempty(current_layer_nodes)
            break
        end
    end
end

function find_next_layer_region_info(Dlw_active_prev, glw_active_prev, Alw_prev, clw_prev, Wl, bl, layer_nr)
    # 1. Get interior point (Fallback mechanism kept for stability)
    if layer_nr != 1
        x = get_interior_point_adaptive(Dlw_active_prev, glw_active_prev)

        is_valid = false
        if x !== nothing && !any(isnan, x)
            if isempty(Dlw_active_prev)
                is_valid = true
            else
                if maximum(Dlw_active_prev * x .- glw_active_prev) <= 1e-5
                    is_valid = true
                end
            end
        end

        # GD Fallback
        if !is_valid
            dim = size(Alw_prev, 2)
            x_curr = randn(dim)
            lr = 0.05; decay = 0.99; max_iter = 500
            for _ in 1:max_iter
                if isempty(Dlw_active_prev); is_valid = true; x = x_curr; break; end
                violations = Dlw_active_prev * x_curr .- glw_active_prev
                mask = violations .> 1e-7
                if !any(mask); x = x_curr; is_valid = true; break; end
                grad = (Dlw_active_prev[mask, :])' * violations[mask]
                gnorm = norm(grad)
                if gnorm > 1e-8; x_curr .-= (lr * grad) / gnorm; else; x_curr = randn(dim); end
                lr *= decay
            end
        end

        if !is_valid
            println("Region not valid")
            return Dict{Vector{Int}, Dict}()
        end
    else
        x = rand(size(Alw_prev, 2))
    end

    # 2. Initial Activation
    z = Wl * Alw_prev * x + Wl * clw_prev + bl
    q0 = Int.(z .> 0)

    traversed = Dict{Vector{Int}, Dict}()
    queue = [q0]

    while !isempty(queue)
        q = popfirst!(queue)

        Dlw, glw, Alw, clw = calculate_next_layer_quantities(Wl, bl, q, Alw_prev, clw_prev)

        # Use simple LP to find active constraints first (fast filter)
        qi_act, _ = find_active_indices_fast(Dlw, glw, Dlw_active_prev, glw_active_prev)

        D_local_active = Dlw[qi_act, :]
        g_local_active = glw[qi_act]
        D_total = vcat(D_local_active, Dlw_active_prev)
        g_total = vcat(g_local_active, glw_active_prev)

        # --- NEW: EXACT GEOMETRY CHECK via CDDLib ---
        # This replaces the ambiguity of LP solvers with exact V-rep conversion
        is_bounded, volume_val, is_feasible = get_exact_geometry(D_total, g_total)

        if is_feasible
            traversed[q] = Dict(
                "q_tilde" => qi_act,
                "bounded" => is_bounded,
                "volume"  => volume_val,
                "Dlw" => Dlw,
                "glw" => glw,
                "Alw" => Alw,
                "clw" => clw,
                "Dlw_active" => D_local_active,
                "glw_active" => g_local_active
            )

            for i_act in qi_act
                q_new = copy(q)
                q_new[i_act] = 1 - q_new[i_act]
                if !haskey(traversed, q_new) && !(q_new in queue)
                     push!(queue, q_new)
                end
            end
        end
    end
    return traversed
end

end # module
