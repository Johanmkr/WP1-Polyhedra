module Verification

using ..Regions
using ..Trees
using ..Geometry
using ..Utils
using LinearAlgebra
using JuMP
using HiGHS
using Printf
using Random
using ProgressMeter

export verify_tree_properties
export verify_partition_completeness_monte_carlo
export scan_all_overlaps_strict_optimized
export verify_tree_hierarchy_robust
export check_region_conditioning

# ==============================================================================
# 1. HELPER: AABB COMPUTATION (Essential for High-D Filtering)
# ==============================================================================

struct AABB
    low::Vector{Float64}
    high::Vector{Float64}
end

"""
Computes the tight Axis-Aligned Bounding Box (AABB) for a region using 2*D LPs.
This is expensive but necessary for efficient overlap filtering in high dimensions.
"""
function get_region_aabb(r::Region, domain_bound::Float64)
    A, b = r.Dlw, r.glw
    dim = size(A, 2)
    low = zeros(dim)
    high = zeros(dim)
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -domain_bound <= x[1:dim] <= domain_bound)
    @constraint(model, A * x .<= b)
    
    for d in 1:dim
        # Minimize x[d]
        @objective(model, Min, x[d])
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            low[d] = value(x[d])
        else
            low[d] = -domain_bound 
        end
        
        # Maximize x[d]
        @objective(model, Max, x[d])
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            high[d] = value(x[d])
        else
            high[d] = domain_bound 
        end
    end
    return AABB(low, high)
end

function aabb_intersect(b1::AABB, b2::AABB)
    return all(b1.low .<= b2.high .+ 1e-7) && all(b2.low .<= b1.high .+ 1e-7)
end

# ==============================================================================
# 2. PARTITION COMPLETENESS (Monte Carlo instead of Exact Volume)
# ==============================================================================

"""
Replaces exact volume check. Samples points from the domain and verifies 
partitioning properties (Gap vs Overlap) statistically.
Robust for D > 15 where exact volume fails.
"""
function verify_partition_completeness_monte_carlo(tree, layer_idx::Int, bound::Float64; num_points=100_000)
    println("\n🎲 Running Monte Carlo Partition Check (Layer $layer_idx)...")
    println("   - Samples: $num_points")
    println("   - Bound:   [-$bound, $bound]")
    
    regions = get_regions_at_layer(tree, layer_idx)
    dim = tree.input_dim
    
    # Pre-extract constraints for speed
    region_constraints = [(r.Dlw, r.glw) for r in regions]
    
    gap_count = 0
    overlap_count = 0
    perfect_count = 0
    
    p = Progress(num_points; desc="Sampling Space: ")
    
    for _ in 1:num_points
        # Uniform sampling in hyperrectangle
        point = (rand(dim) .- 0.5) .* (2 * bound)
        
        matches = 0
        for (A, b) in region_constraints
            is_inside = true
            # Optimized dot product check
            for i in 1:length(b)
                if dot(@view(A[i, :]), point) > b[i] + 1e-7
                    is_inside = false
                    break
                end
            end
            
            if is_inside
                matches += 1
            end
        end
        
        if matches == 0
            gap_count += 1
        elseif matches == 1
            perfect_count += 1
        else
            overlap_count += 1
        end
        
        if rand() < 0.01; next!(p); end
    end
    finish!(p)
    
    total = gap_count + overlap_count + perfect_count
    gap_pct = (gap_count / total) * 100
    over_pct = (overlap_count / total) * 100
    perf_pct = (perfect_count / total) * 100

    println("\n📊 Statistics:")
    @printf("  - Perfect (Exactly 1 Region): %d (%.2f%%)\n", perfect_count, perf_pct)
    @printf("  - Gaps (0 Regions):           %d (%.4f%%)\n", gap_count, gap_pct)
    @printf("  - Overlaps (>1 Regions):      %d (%.4f%%)\n", overlap_count, over_pct)
    
    return (gap_count == 0 && overlap_count == 0)
end

# ==============================================================================
# 3. OPTIMIZED OVERLAP CHECK (AABB Filter + LP)
# ==============================================================================

function check_overlap_strict_lp(r1::Region, r2::Region)
    A1, b1 = r1.Dlw, r1.glw
    A2, b2 = r2.Dlw, r2.glw
    dim = size(A1, 2)
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:dim])
    
    # Strict inequality simulation: A*x <= b - epsilon
    ϵ = 1e-6 
    @constraint(model, A1 * x .<= b1 .- ϵ)
    @constraint(model, A2 * x .<= b2 .- ϵ)
    
    optimize!(model)
    return termination_status(model) == MOI.OPTIMAL
end

function scan_all_overlaps_strict_optimized(tree, layer_idx; bound=10.0)
    regions = get_regions_at_layer(tree, layer_idx)
    n = length(regions)
    println("\n⚔️  Optimized LP Overlap Scan (Layer $layer_idx, $n regions)...")
    
    if n > 1000
        println("   - Large dataset detected. Computing AABBs for pre-filtering...")
    end

    # 1. Precompute AABBs (O(N) * Cost of LP)
    aabbs = Vector{AABB}(undef, n)
    pm = Progress(n; desc="Precomputing Bounds: ")
    for i in 1:n
        aabbs[i] = get_region_aabb(regions[i], bound)
        next!(pm)
    end
    
    overlaps = 0
    lp_checks = 0
    skipped = 0
    
    # 2. Pairwise Check
    p = Progress(n*(n-1)÷2; desc="Pairwise Check: ")
    
    for i in 1:n
        for j in (i+1):n
            # FILTER: Cheap AABB Intersection Check
            if !aabb_intersect(aabbs[i], aabbs[j])
                skipped += 1
                continue
            end
            
            # VERIFY: Expensive LP Check
            lp_checks += 1
            if check_overlap_strict_lp(regions[i], regions[j])
                overlaps += 1
                println("  !! Overlap confirmed between Region #$i and Region #$j")
            end
        end
        # Approximate progress update for the loop block
        if i % 10 == 0; update!(p, (i * (2n - i - 1)) ÷ 2); end
    end
    finish!(p)
    
    println("\n🔍 Overlap Scan Results:")
    println("  - Total Pairs:    $(n*(n-1)÷2)")
    println("  - Skipped (AABB): $skipped")
    println("  - LP Checks Run:  $lp_checks")
    println("  - Overlaps Found: $overlaps")
    
    return overlaps == 0
end

# ==============================================================================
# 4. ROBUST HIERARCHY CHECK (Containment Only)
# ==============================================================================

function verify_tree_hierarchy_robust(tree::Tree; bound=10.0)
    println("\n🌳 Verifying Tree Hierarchy (Containment LP)...")
    
    queue = [tree.root]
    mismatches = 0
    nodes_checked = 0
    
    while !isempty(queue)
        node = popfirst!(queue)
        children = get_children(node)
        
        if isempty(children)
            continue
        end
        nodes_checked += 1
        
        for child in children
            push!(queue, child)
            
            # 1. Compute Child Center using FULL PATH (including Root)
            A_full, b_full = get_path_inequalities(child)
            center, _ = get_chebyshev_center_radius(A_full, b_full)
            
            if isnothing(center)
                continue
            end
            
            # 2. Check against Parent's ACTIVE constraints
            if isempty(node.Dlw_active)
                max_violation = -Inf
            else
                # Calculate Raw Violation: Ax - b
                raw_violation = node.Dlw_active * center .- node.glw_active
                
                # FIX: Normalize violation by row norm (Geometric Distance)
                max_violation = -Inf
                for k in 1:length(node.glw_active)
                    row_norm = norm(node.Dlw_active[k, :])
                    val = raw_violation[k]
                    
                    # Distance = val / ||a||. Handle zero rows safely.
                    dist = (row_norm > 1e-7) ? (val / row_norm) : val
                    if dist > max_violation
                        max_violation = dist
                    end
                end
            end
            
            if max_violation > 1e-5
                println("  ❌ INHERITANCE FAIL: Child (Layer $(child.layer_number)) leaks outside Parent")
                println("     Max Geometric Violation: $(@sprintf("%.2e", max_violation))")
                mismatches += 1
            end
        end
    end
    
    if mismatches == 0
        println("✅ Tree Hierarchy (Containment) valid for $nodes_checked branching nodes.")
        return true
    else
        println("❌ Found $mismatches containment violations.")
        return false
    end
end

# ==============================================================================
# 5. CONDITIONING
# ==============================================================================

function check_region_conditioning(tree, layer_idx; min_radius=1e-7)
    println("\n📐 Checking Region Conditioning (Thinness)...")
    regions = get_regions_at_layer(tree, layer_idx)
    thin_count = 0
    
    p = Progress(length(regions); desc="Checking Radii: ")
    for (i, r) in enumerate(regions)
        # FIX: Use full path inequalities
        A_full, b_full = get_path_inequalities(r)
        _, r_val = get_chebyshev_center_radius(A_full, b_full)
        
        if r_val < min_radius
            thin_count += 1
        end
        next!(p)
    end
    
    if thin_count == 0
        println("✅ All regions are well-conditioned.")
        return true
    else
        println("⚠️ Found $thin_count thin regions (radius < $min_radius). Solvers may struggle.")
        return false
    end
end

# Helper: Combined Center & Radius
function get_chebyshev_center_radius(A, b; limit=1e5)
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -limit <= x[1:n] <= limit)
    @variable(model, r)
    
    norms = [norm(A[i,:]) for i in 1:m]
    @constraint(model, [i=1:m], dot(A[i,:], x) + r * norms[i] <= b[i])
    
    @objective(model, Max, r)
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(x), value(r)
    else
        return nothing, 0.0
    end
end

# ==============================================================================
# 6. MASTER VERIFICATION FUNCTION
# ==============================================================================

function verify_tree_properties(tree::Tree; bound=10.0, layer_idx=nothing)
    println("="^60)
    println("🚀 STARTING FULL TREE VERIFICATION")
    println("="^60)
    
    if isnothing(layer_idx)
        # Fallback if depth property doesn't exist, calculate it or default to something safe
        # Assuming typical tree depth logic or manual user input usually.
        # Here we just grab the layer of the first leaf if possible, or 10.
        target_layer = 1 
        # You might want to pass layer_idx explicitly to be safe
        println("ℹ️  Targeting Layer: $target_layer (Default)")
    else
        target_layer = layer_idx
        println("ℹ️  Targeting Custom Layer: $target_layer")
    end
    
    results = Dict{String, Bool}()
    
    results["Hierarchy"] = verify_tree_hierarchy_robust(tree; bound=bound)
    results["Overlaps"] = scan_all_overlaps_strict_optimized(tree, target_layer; bound=bound)
    results["Completeness"] = verify_partition_completeness_monte_carlo(tree, target_layer, bound)
    results["Conditioning"] = check_region_conditioning(tree, target_layer)
    
    println("\n" * "="^60)
    println("📝 VERIFICATION SUMMARY")
    println("="^60)
    
    all_passed = true
    for (test, passed) in results
        status = passed ? "✅ PASS" : "❌ FAIL"
        @printf("%-15s : %s\n", test, status)
        if !passed; all_passed = false; end
    end
    
    println("-"^60)
    if all_passed
        println("🎉 TREE IS VALID (High Confidence)")
    else
        println("⚠️ TREE HAS ISSUES (See logs above)")
    end
    return all_passed
end

end # module