module Verification

using ..Regions
using ..Trees
using ..Geometry
using ..Utils
using LinearAlgebra
using JuMP
using HiGHS
using Polyhedra
using CDDLib
using ProgressMeter
using Printf

export verify_volume_conservation, check_point_partition, scan_all_overlaps_strict



# ==============================================================================
# 2. PARTITION CHECK (Volume Conservation)
# ==============================================================================

function verify_volume_conservation(tree, layer_idx::Int, bound::Float64; tol=1e-5)
    println("\n🔍 Checking Volume Conservation for Layer $layer_idx within [-$(bound), $(bound)]...")
    
    # Force GC to clean up any potential Python objects before heavy lifting
    GC.gc()
    
    regions = get_regions_at_layer(tree, layer_idx)
    dim = tree.input_dim
    theo_vol = (2 * bound)^dim
    
    total_region_vol = 0.0
    
    # Single-threaded loop to avoid PyCall GIL crashes
    # p = Progress(length(regions); desc="Computing Volumes: ")
    
    for r in regions
        vol = get_region_volume(r, bound=bound)
        total_region_vol += vol
        # next!(p)
    end
    
    diff = total_region_vol - theo_vol
    
    @printf("\n  - Theoretical Box Volume: %.4f\n", theo_vol)
    @printf("  - Sum of Region Volumes:  %.4f\n", total_region_vol)
    @printf("  - Difference:             %.4e\n", diff)
    
    if abs(diff) < tol
        println("✅ PASSED: Regions partition the space perfectly (within tolerance).")
        return true
    elseif diff > tol
        println("❌ FAILED: Sum > Box. Overlaps exist.")
        return false
    else
        println("❌ FAILED: Sum < Box. Holes (missing space) exist.")
        return false
    end
end

# ==============================================================================
# 3. POINT-WISE EXCLUSIVITY CHECK
# ==============================================================================

function check_point_partition(tree, layer_idx::Int, num_points::Int; bound=10.0)
    println("\n🎯 Running Point-wise Monte Carlo Check ($num_points points)...")
    
    regions = get_regions_at_layer(tree, layer_idx)
    dim = tree.input_dim
    
    region_constraints = []
    for r in regions
        push!(region_constraints, get_path_inequalities(r))
    end
    
    overlap_errors = 0
    gap_errors = 0
    
    # Kept single-threaded for safety
    for k in 1:num_points
        x = (rand(dim) .- 0.5) .* (2 * bound)
        
        containment_count = 0
        
        for (A, b) in region_constraints
            if all(A * x .<= b .+ 1e-7)
                containment_count += 1
            end
        end
        
        if containment_count == 0
            gap_errors += 1
        elseif containment_count > 1
            overlap_errors += 1
        end
    end
    
    if overlap_errors == 0 && gap_errors == 0
        println("✅ PASSED: All sampled points fall into exactly 1 region.")
    else
        println("❌ FAILED:")
        println("   - Points in 0 regions (Holes):    $gap_errors")
        println("   - Points in >1 regions (Overlap): $overlap_errors")
    end
end

# ==============================================================================
# 4. STRICT PAIRWISE OVERLAP CHECK (LP Solver)
# ==============================================================================

function check_overlap_strict(r1::Region, r2::Region)
    A1, b1 = get_path_inequalities(r1)
    A2, b2 = get_path_inequalities(r2)
    dim = size(A1, 2)
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x[1:dim])
    
    ϵ = 1e-6 
    @constraint(model, A1 * x .<= b1 .- ϵ)
    @constraint(model, A2 * x .<= b2 .- ϵ)
    
    optimize!(model)
    return termination_status(model) == MOI.OPTIMAL
end

function scan_all_overlaps_strict(tree, layer_idx)
    regions = get_regions_at_layer(tree, layer_idx)
    n = length(regions)
    println("\n⚔️  Strict LP Overlap Scan (Layer $layer_idx, $n regions)...")
    
    overlaps = 0
    p = Progress(n*(n-1)÷2)
    
    for i in 1:n
        for j in (i+1):n
            if check_overlap_strict(regions[i], regions[j])
                overlaps += 1
                println("  !! Overlap confirmed between R$i and R$j")
            end
            next!(p)
        end
    end
    
    if overlaps == 0
        println("✅ No strict overlaps found.")
    else
        println("❌ Found $overlaps overlapping pairs.")
    end
end

function verify_tree_hierarchy(tree::Tree; tol=1e-5, bound=10)
    println("\n🌳 Verifying Tree Hierarchy (Parent vs. Children)...")
    
    # We traverse the tree (BFS or simple iteration if nodes are stored linearly)
    # Assuming tree.nodes is accessible or we walk from root
    
    queue = [tree.root]
    mismatches = 0
    
    while !isempty(queue)
        node = popfirst!(queue)
        children = get_children(tree, node) # You likely have a helper for this
        
        if isempty(children)
            continue
        end
        
        # 1. Check Constraint Inheritance
        # Every child must satisfy parent's constraints
        parent_poly = get_bounded_polyhedron(node, bound)
        parent_vol = volume(parent_poly)
        
        child_vol_sum = 0.0
        
        for child in children
            # Add to queue for next layer
            push!(queue, child)
            
            child_poly = get_bounded_polyhedron(child, bound)
            if isnothing(child_poly)
                println("  ⚠️ Child of Node $(node.id) is degenerate/empty.")
                continue
            end
            
            # Volume accumulation
            child_vol_sum += volume(child_poly)
            
            # Strict containment check (Sample point)
            # A quick check: The center of the child must be inside the parent
            child_center = get_chebyshev_center(child.Dlw, child.glw) # Uses your robust function
            if !isnothing(child_center)
                # Check A_parent * x <= b_parent
                violation = node.Dlw * child_center .- node.glw
                if maximum(violation) > 1e-6
                    println("  ❌ INHERITANCE FAIL: Child $(child.id) leaks outside Parent $(node.id)")
                    mismatches += 1
                end
            end
        end
        
        # 2. Local Volume Conservation
        diff = abs(parent_vol - child_vol_sum)
        if diff > tol
            println("  ❌ PARTITION FAIL: Node $(node.id) volume mismatch.")
            println("     Parent: $parent_vol, Children Sum: $child_vol_sum, Diff: $diff")
            mismatches += 1
        end
    end
    
    if mismatches == 0
        println("✅ Tree Hierarchy strictly valid.")
    else
        println("❌ Found $mismatches structural violations.")
    end
end

function check_region_conditioning(tree, layer_idx; min_radius=1e-7)
    println("\n📐 Checking Region Conditioning (Thinness)...")
    regions = get_regions_at_layer(tree, layer_idx)
    thin_count = 0
    
    for r in regions
        # We reuse your robust interior point code, but we extract the radius
        # You might need to modify 'get_feasible_point' to return (x, radius) or call a new helper
        
        # Quick implementation of radius check:
        r_val = get_chebyshev_radius(r.Dlw, r.glw)
        
        if r_val < min_radius
            println("  ⚠️ Region $(r.id) is extremely thin! (Radius: $r_val)")
            thin_count += 1
        end
    end
    
    if thin_count == 0
        println("✅ All regions are well-conditioned (fat enough).")
    else
        println("⚠️ Found $thin_count thin regions. Solvers may struggle here.")
    end
end

# Helper for conditioning check
function get_chebyshev_radius(A, b; limit=1e5)
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -limit <= x[1:n] <= limit)
    @variable(model, r)
    for i in 1:m
        @constraint(model, dot(A[i,:], x) + r * norm(A[i,:]) <= b[i])
    end
    @objective(model, Max, r)
    optimize!(model)
    return value(r)
end

end # module
