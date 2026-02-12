module Geometry

using LinearAlgebra
using JuMP
using HiGHS
using Polyhedra
using CDDLib

export find_active_indices_exact, get_feasible_point, find_active_indices_lp

using LinearAlgebra
using Polyhedra
using CDDLib

function find_active_indices_exact(D_local, g_local, D_prev, g_prev)
    # 1. Combine all constraints
    D_all = vcat(D_local, D_prev)
    g_all = vcat(g_local, g_prev)
    n_local = size(D_local, 1)

    # 2. Create the Polyhedron using CDDLib
    h = hrep(D_all, g_all)
    poly = polyhedron(h, CDDLib.Library(:exact))

    # 3. Remove Redundancy (H-Rep level)
    removehredundancy!(poly)

    # 4. Check Boundedness & Volume (V-Rep level)
    # We force the conversion to V-representation (vertices + rays).
    # This is the expensive step (Vertex Enumeration).
    v = vrep(poly)
    
    # Robust boundedness check:
    # A polyhedron is bounded iff it has 0 rays.
    # We check nrays() if available, or check if the rays iterator is empty.
    is_bnd = false
    if isdefined(Polyhedra, :nrays)
        is_bnd = (Polyhedra.nrays(v) == 0)
    else
        # Fallback if nrays is not exported
        is_bnd = isempty(Polyhedra.rays(v))
    end
    
    vol = 0.0
    if !is_bnd
        vol = Inf
    else
        # If bounded, we can calculate the volume safely
        vol = Polyhedra.volume(poly)
    end

    # 5. Extract Minimal Constraints
    minimal_hrep = hrep(poly)
    active_indices = Int[]
    
    # Helper to check geometric equivalence
    function is_equivalent(a1, b1, a2, b2; tol=1e-6)
        n1 = norm(a1)
        n2 = norm(a2)
        if n1 < 1e-8 || n2 < 1e-8; return false; end
        
        an1, bn1 = a1 ./ n1, b1 / n1
        an2, bn2 = a2 ./ n2, b2 / n2
        
        return norm(an1 - an2) < tol && abs(bn1 - bn2) < tol
    end

    # 6. Map back to original indices
    for i in 1:n_local
        my_a = D_local[i, :]
        my_b = g_local[i]
        
        found_match = false
        for h in allhalfspaces(minimal_hrep)
            if is_equivalent(my_a, my_b, h.a, h.β)
                found_match = true
                break
            end
        end
        
        if found_match
            push!(active_indices, i)
        end
    end

    return active_indices, is_bnd, vol
end


function find_active_indices_lp(D_local, g_local, D_prev, g_prev; model=nothing, tol=1e-9)
    # 1. Combine all constraints
    D_all = vcat(D_local, D_prev)
    g_all = vcat(g_local, g_prev)
    
    n_local = length(g_local)
    n_total = length(g_all)
    dim = size(D_all, 2)
    
    # 2. Normalize constraints to handle scaling (a'x <= b vs 2a'x <= 2b)
    # We store normalized versions to detect duplicates and for the LP
    D_norm = zeros(size(D_all))
    g_norm = zeros(n_total)
    
    for i in 1:n_total
        nm = norm(D_all[i, :])
        if nm < 1e-9 # Handle zero rows if any
            D_norm[i, :] = D_all[i, :]
            g_norm[i] = g_all[i]
        else
            D_norm[i, :] = D_all[i, :] ./ nm
            g_norm[i] = g_all[i] / nm
        end
    end

    # 3. Deduplicate Constraints
    # We map every constraint index to a "unique representative" index.
    # If D_local has duplicates of an active constraint, we want to keep ALL of them.
    # LP redundancy checks will fail if we include a duplicate in the constraint set
    # (Constraint A will be redundant b/c of A', and A' redundant b/c of A).
    
    unique_indices = Int[] # Indices of unique constraints to keep in the LP
    map_to_unique = zeros(Int, n_total) # Maps i -> unique_rep_index
    
    # Simple N^2 deduplication (sufficient for typical constraint counts)
    # For massive sets, use a spatial hash or KDTree.
    for i in 1:n_total
        is_dup = false
        for u in unique_indices
            # Check parallel direction and same bound
            if norm(D_norm[i, :] - D_norm[u, :]) < tol && abs(g_norm[i] - g_norm[u]) < tol
                map_to_unique[i] = u
                is_dup = true
                break
            end
        end
        if !is_dup
            push!(unique_indices, i)
            map_to_unique[i] = i
        end
    end

    # 4. Prepare the LP Solver
    if isnothing(model)
        model = Model(HiGHS.Optimizer)
        set_silent(model)
    else
        empty!(model)
        set_silent(model)
    end
    @variable(model, x[1:dim])
    
    # We add ALL unique constraints initially
    # We will modify bounds or relaxations later, or simpler: 
    # Just rebuild/update constraints. For speed, we add all and use a loop to relax one.
    # However, deleting/adding constraints in a loop can be slow. 
    # A common efficient approach is adding all constraints and using "dual simplex" 
    # but re-solving N times is unavoidable.
    
    # Store references to constraints
    con_refs = Dict{Int, ConstraintRef}()
    for u in unique_indices
        con_refs[u] = @constraint(model, dot(D_all[u, :], x) <= g_all[u])
    end

    active_unique_reps = Set{Int}()

    # 5. Check Redundancy for unique constraints that represent at least one local constraint
    # We only care if a unique constraint is active if it maps back to a local index.
    
    # Identify unique reps that cover at least one local index
    relevant_reps = Set(map_to_unique[1:n_local])
    
    for u in relevant_reps
        # --- The Redundancy Check ---
        
        # A. Disable the constraint we are testing
        delete(model, con_refs[u])
        
        # B. Maximize in the direction of the constraint normal
        @objective(model, Max, dot(D_all[u, :], x))
        
        optimize!(model)
        
        is_active = false
        status = termination_status(model)
        
        if status == MOI.OPTIMAL
            obj_val = objective_value(model)
            # If we can go strictly past the bound, the constraint was blocking us.
            if obj_val > g_all[u] + tol
                is_active = true
            end
        elseif status == MOI.DUAL_INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            # Unbounded means the region opens up infinitely without this constraint
            is_active = true
        else
            # If the remaining set is Infeasible, the geometry is broken or empty.
            # Usually implies all constraints are 'active' in defining the emptiness,
            # or inputs are bad. Assuming feasible region for now.
        end
        
        if is_active
            push!(active_unique_reps, u)
        end
        
        # C. Re-add the constraint for the next iteration
        con_refs[u] = @constraint(model, dot(D_all[u, :], x) <= g_all[u])
    end

    # 6. Map back to original local indices
    active_indices = Int[]
    for i in 1:n_local
        rep = map_to_unique[i]
        if rep in active_unique_reps
            push!(active_indices, i)
        end
    end
    
    return active_indices, true
end


# --- LP Solvers (Unchanged helpers) ---

function get_feasible_point(A::Matrix{Float64}, b::Vector{Float64}; model=nothing, limit=1e5)
    # 1. Model Management
    if isnothing(model)
        model = Model(HiGHS.Optimizer)
        set_silent(model)
    else
        empty!(model) # Clears variables and constraints
        # Re-attach optimizer if empty! detached it (JuMP < 1.0 specific, but safe to keep silent)
        set_silent(model) 
    end

    m, n = size(A)
    
    # 2. Re-populate Model
    # Note: creating variables every time is fast.
    @variable(model, -limit <= x[1:n] <= limit)
    @variable(model, r <= limit) 

    # 3. Add Constraints
    # We loop to add them. For maximum speed with fixed sizes, one could reuse 
    # constraint references, but since 'm' changes, rebuilding is best.
    for i in 1:m
        a_row = A[i, :]
        a_norm = norm(a_row)
        
        if a_norm > 1e-8
            @constraint(model, dot(a_row, x) + a_norm * r <= b[i])
        elseif b[i] < -1e-8
            return nothing
        end
    end

    @objective(model, Max, r)
    optimize!(model)

    st = termination_status(model)
    if st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        max_r = value(r)
        if max_r >= -1e-7
            return value.(x)
        end
    end
    
    return nothing
end

end # module
