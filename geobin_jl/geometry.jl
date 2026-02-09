# Geometry functions

function calculate_next_layer_quantities(Wl, bl, qlw, Alw_prev, clw_prev)
    Wl_hat = Wl * Alw_prev
    bl_hat = Wl * clw_prev + bl
    s_vec = -2.0 .* qlw .+ 1.0
    Dlw = s_vec .* Wl_hat
    glw = -(s_vec .* bl_hat)
    Alw = qlw .* Wl_hat
    clw = qlw .* bl_hat
    return Dlw, glw, Alw, clw
end

function get_exact_geometry(D, g)
    # 1. Create H-representation
    # We map to the library inside the hrep call to ensure type consistency
    h = Polyhedra.hrep(D, g)

    # 2. Use EXACT Arithmetic (Rational)
    # :float is fast but crashes/fails on thin NN polytopes.
    # :exact uses GMP (BigInt/Rational) to guarantee correctness.
    lib = CDDLib.Library(:exact)

    try
        # Create polyhedron
        poly = Polyhedra.polyhedron(h, lib)

        # 3. Force V-Rep computation
        # Note: In exact mode, this might take longer but won't crash on slivers
        v = Polyhedra.vrep(poly)

        # 4. Feasibility Check
        # If it has 0 points and 0 rays, it is truly empty.
        if Polyhedra.npoints(v) == 0 && Polyhedra.nrays(v) == 0
            return (true, 0.0, false)
        end

        # 5. Unbounded Check
        if Polyhedra.nrays(v) > 0
            return (false, Inf, true)
        end

        # 6. Degeneracy Check (Volume)
        # ONLY use this check if you are in low dimensions (d < 5).
        # Otherwise, trust existence of vertices.
        # Calculating exact volume in high-dim is also very expensive.

        # Optimization: Skip volume calc for validity check.
        # If npoints > 0 and bounded, it is a valid polytope.
        return (true, 1.0, true) # Return dummy volume 1.0 to indicate "Valid/Finite"

    catch e
        # Log the error so you know if you are losing regions
        println("\n!!! Geometry Check Error: $e")
        println("Constraints D shape: $(size(D))")
        return (true, 0.0, false)
    end
end

# --- KEEPING THE FAST LP FILTER FOR INITIAL ACTIVE SET ---
function find_active_indices_fast(D_local, g_local, D_prev, g_prev; tol=1e-5)
    D_all = vcat(D_local, D_prev)
    g_all = vcat(g_local, g_prev)
    n_local = size(D_local, 1)
    dim = size(D_all, 2)
    active_indices = Int[]

    model = direct_model(HiGHS.Optimizer())
    set_silent(model)
    set_attribute(model, "presolve", "off")
    @variable(model, x[1:dim])
    @constraint(model, cons[i=1:size(D_all,1)], dot(D_all[i,:], x) <= g_all[i])

    for i in 1:n_local
        d_row = D_all[i, :]
        if norm(d_row) < 1e-10; continue; end

        original_rhs = g_all[i]
        set_normalized_rhs(cons[i], 1e20)
        @objective(model, Max, dot(d_row, x))
        optimize!(model)

        st = termination_status(model)
        is_active = false
        if st == MOI.DUAL_INFEASIBLE
            is_active = true
        elseif st == MOI.OPTIMAL
            if objective_value(model) > original_rhs + tol
                is_active = true
            end
        end

        if is_active
            push!(active_indices, i)
            set_normalized_rhs(cons[i], original_rhs)
        end
    end
    return active_indices, true # Boundedness is checked later by CDDLib
end

# --- LP Solvers (Unchanged helpers) ---

function get_interior_point_adaptive(A::Matrix{Float64}, b::Vector{Float64})
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    limit = 1e6
    @variable(model, -limit <= x[i=1:n] <= limit)
    ϵ = 1e-7
    for i in 1:m
        @constraint(model, sum(A[i, j] * x[j] for j in 1:n) <= b[i] - ϵ)
    end
    @objective(model, Min, 0)
    optimize!(model)
    st = termination_status(model)
    if st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)
        return value.(x)
    elseif st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED, MOI.SLOW_PROGRESS)
        val = find_any_feasible_point(A, b, limit)
        return any(isnan, val) ? nothing : val
    else
        return nothing
    end
end

function find_any_feasible_point(A, b, limit)
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -limit <= x[1:n] <= limit)
    @constraint(model, A * x .<= b)
    @objective(model, Min, 0)
    optimize!(model)
    if termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        return value.(x)
    else
        return fill(NaN, size(A, 2))
    end
end
