module Geometry

using LinearAlgebra
using JuMP
using HiGHS
using Polyhedra
using CDDLib

export find_active_indices_exact, get_feasible_point


function find_active_indices_exact(D_local, g_local, D_prev, g_prev)
    # 1. Combine all constraints
    D_all = vcat(D_local, D_prev)
    g_all = vcat(g_local, g_prev)
    n_local = size(D_local, 1)

    # 2. Create the Polyhedron using CDDLib
    # CDDLib is an exact vertex enumeration library. 
    # It handles degeneracy better than standard LP solvers.
    h = hrep(D_all, g_all)
    poly = polyhedron(h, CDDLib.Library(:exact))

    # 3. Remove Redundancy
    # This modifies 'poly' in-place to the minimal H-representation.
    # It solves the exact geometric problem you were trying to approximate.
    removehredundancy!(poly)

    # 4. Extract the Minimal Constraints
    # We get the list of halfspaces (a'x <= b) that survived.
    minimal_hrep = hrep(poly)
    
    active_indices = Int[]
    
    # Helper to check if two constraints are geometrically equivalent
    # We normalize (a, b) -> (a, b) / ||a|| to handle scaling differences
    # e.g., 2x <= 2 is the same as x <= 1
    function is_equivalent(a1, b1, a2, b2; tol=1e-6)
        n1 = norm(a1)
        n2 = norm(a2)
        
        # If one is zero-vector (shouldn't happen in minimal rep), skip
        if n1 < 1e-8 || n2 < 1e-8; return false; end
        
        # Normalize
        an1, bn1 = a1 ./ n1, b1 / n1
        an2, bn2 = a2 ./ n2, b2 / n2
        
        # Check collinearity (dot product ~ 1) and same bound
        # We check norm(diff) to ensure direction is identical, not opposite
        return norm(an1 - an2) < tol && abs(bn1 - bn2) < tol
    end

    # 5. Map back to original indices
    # We iterate through our LOCAL constraints and check if they exist 
    # in the minimal representation.
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
    # println(isempty(active_indices))
    return active_indices, true
end

# --- LP Solvers (Unchanged helpers) ---

function get_feasible_point(A::Matrix{Float64}, b::Vector{Float64}; limit=1e5)
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # 1. Bounded variables to handle unbounded regions safely
    @variable(model, -limit <= x[1:n] <= limit)

    # 2. Variable Radius 'r' ( Geometric Slack )
    # We bound r to avoid unbounded objectives if the region is open
    @variable(model, r <= limit) 

    # 3. Normalized Constraints
    # Algebraic: a_i' * x + s <= b_i
    # Geometric: a_i' * x + ||a_i|| * r <= b_i
    # This ensures 'r' represents the true Euclidean distance to the boundary.
    for i in 1:m
        a_row = A[i, :]
        a_norm = norm(a_row)
        
        if a_norm > 1e-8
            # Normal case: impose geometric padding
            @constraint(model, dot(a_row, x) + a_norm * r <= b[i])
        else
            # Degenerate row (0 vector): just check feasibility
            if b[i] < -1e-8
                return nothing # 0 <= negative is impossible
            end
            # If 0 <= positive, the constraint is trivial and ignored.
        end
    end

    # 4. Objective: Maximize the radius of the inscribed ball
    @objective(model, Max, r)

    optimize!(model)

    # 5. Robust check
    st = termination_status(model)
    if st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        max_r = value(r)
        
        # We accept:
        #  r > 0: Strict interior point
        #  r ≈ 0: Flat region (e.g., a line in 2D), but valid
        # We use a small negative tolerance to allow for floating point noise on flat regions.
        if max_r >= -1e-7
            return value.(x)
        end
    end
    
    return nothing
end

end # module
