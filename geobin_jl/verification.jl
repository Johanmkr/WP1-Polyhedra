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
using Base.Threads 

export verify_tree_properties
export verify_partition_completeness_monte_carlo
export scan_all_overlaps_strict_optimized
export verify_tree_hierarchy_robust
export check_region_conditioning

# ==============================================================================
# 1. HELPER: AABB COMPUTATION
# ==============================================================================

struct AABB
    low::Vector{Float64}
    high::Vector{Float64}
end

function get_region_aabb(r::Region, domain_bound::Float64)
    A, b = get_path_inequalities(r)
    dim = size(A, 2)
    low = zeros(dim)
    high = zeros(dim)
    
    if size(A, 1) == 0
        return AABB(fill(-domain_bound, dim), fill(domain_bound, dim))
    end
    
    # Use a local model for thread safety
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -domain_bound <= x[1:dim] <= domain_bound)
    @constraint(model, A * x .<= b)
    
    for d in 1:dim
        @objective(model, Min, x[d])
        optimize!(model)
        low[d] = (termination_status(model) == MOI.OPTIMAL) ? value(x[d]) : -domain_bound
        
        @objective(model, Max, x[d])
        optimize!(model)
        high[d] = (termination_status(model) == MOI.OPTIMAL) ? value(x[d]) : domain_bound
    end
    return AABB(low, high)
end

function aabb_intersect(b1::AABB, b2::AABB)
    return all(b1.low .<= b2.high .+ 1e-7) && all(b2.low .<= b1.high .+ 1e-7)
end

# ==============================================================================
# 2. PARTITION COMPLETENESS (PARALLEL MONTE CARLO)
# ==============================================================================

function verify_partition_completeness_monte_carlo(tree, layer_idx::Int, bound::Float64; num_points=100_000)
    println("\n🎲 Running Monte Carlo Partition Check (Layer $layer_idx)...")
    println("   - Samples: $num_points")
    println("   - Threads: $(Threads.nthreads())")
    
    regions = get_regions_at_layer(tree, layer_idx)
    dim = tree.input_dim
    
    # Pre-calculate constraints (Read-only for threads)
    # Using a simple map here; usually fast enough to be serial
    region_constraints = [get_path_inequalities(r) for r in regions]
    
    gap_count = Threads.Atomic{Int}(0)
    overlap_count = Threads.Atomic{Int}(0)
    perfect_count = Threads.Atomic{Int}(0)
    
    # Parallel Sampling
    Threads.@threads for _ in 1:num_points
        # Random point in [-bound, bound]
        point = (rand(dim) .- 0.5) .* (2 * bound)
        matches = 0
        
        for (A, b) in region_constraints
            is_inside = true
            # Optimized check
            for i in 1:length(b)
                if dot(@view(A[i, :]), point) > b[i] + 1e-7
                    is_inside = false; break
                end
            end
            if is_inside; matches += 1; end
        end
        
        if matches == 0
            Threads.atomic_add!(gap_count, 1)
        elseif matches == 1
            Threads.atomic_add!(perfect_count, 1)
        else
            Threads.atomic_add!(overlap_count, 1)
        end
    end
    
    g_val = gap_count[]
    o_val = overlap_count[]
    p_val = perfect_count[]
    total = num_points
    
    println("\n📊 Statistics:")
    @printf("  - Perfect (1 Region): %d (%.2f%%)\n", p_val, (p_val/total)*100)
    @printf("  - Gaps (0 Regions):   %d (%.4f%%)\n", g_val, (g_val/total)*100)
    @printf("  - Overlaps (>1 Reg):  %d (%.4f%%)\n", o_val, (o_val/total)*100)
    
    return (g_val == 0 && o_val == 0)
end

# ==============================================================================
# 3. OPTIMIZED OVERLAP CHECK (Parallel AABB)
# ==============================================================================

function check_overlap_strict_lp(r1::Region, r2::Region)
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

function scan_all_overlaps_strict_optimized(tree, layer_idx; bound=10.0)
    regions = get_regions_at_layer(tree, layer_idx)
    n = length(regions)
    println("\n⚔️  Optimized LP Overlap Scan (Layer $layer_idx, $n regions)...")
    
    # 1. Parallel Precompute AABBs
    println("   - Precomputing AABBs (Parallel)...")
    aabbs = Vector{AABB}(undef, n)
    
    Threads.@threads for i in 1:n
        aabbs[i] = get_region_aabb(regions[i], bound)
    end
    
    overlaps = 0
    skipped = 0
    lp_checks = 0
    
    # 2. Pairwise Check (Serial loop to avoid shared counter race conditions)
    # The heavy lifting was the AABB generation above.
    pm = Progress(n*(n-1)÷2; desc="Pairwise Check: ")
    
    for i in 1:n
        for j in (i+1):n
            if !aabb_intersect(aabbs[i], aabbs[j])
                skipped += 1
                continue
            end
            
            lp_checks += 1
            if check_overlap_strict_lp(regions[i], regions[j])
                overlaps += 1
            end
        end
        if i % 10 == 0; update!(pm, (i * (2n - i - 1)) ÷ 2); end
    end
    finish!(pm)
    
    println("\n🔍 Overlap Scan Results:")
    println("  - Total Pairs:    $(n*(n-1)÷2)")
    println("  - Skipped (AABB): $skipped")
    println("  - LP Checks Run:  $lp_checks")
    println("  - Overlaps Found: $overlaps")
    
    return overlaps == 0
end

# ==============================================================================
# 4. ROBUST HIERARCHY CHECK
# ==============================================================================

function verify_tree_hierarchy_robust(tree::Tree; bound=10.0)
    println("\n🌳 Verifying Tree Hierarchy (Containment LP)...")
    
    queue = [tree.root]
    mismatches = 0
    nodes_checked = 0
    
    while !isempty(queue)
        node = popfirst!(queue)
        children = get_children(node)
        
        if isempty(children); continue; end
        nodes_checked += 1
        
        for child in children
            push!(queue, child)
            A_full, b_full = get_path_inequalities(child)
            center, _ = get_chebyshev_center_radius(A_full, b_full)
            
            if isnothing(center); continue; end
            
            if !isempty(node.Dlw_active)
                raw_violation = node.Dlw_active * center .- node.glw_active
                max_violation = -Inf
                
                for k in 1:length(node.glw_active)
                    row_norm = norm(node.Dlw_active[k, :])
                    val = raw_violation[k]
                    dist = (row_norm > 1e-7) ? (val / row_norm) : val
                    if dist > max_violation; max_violation = dist; end
                end
                
                if max_violation > 1e-5
                    mismatches += 1
                end
            end
        end
    end
    
    if mismatches == 0
        println("✅ Tree Hierarchy valid for $nodes_checked nodes.")
        return true
    else
        println("❌ Found $mismatches containment violations.")
        return false
    end
end

# ==============================================================================
# 5. CONDITIONING (Parallel)
# ==============================================================================

function check_region_conditioning(tree, layer_idx; min_radius=1e-7)
    println("\n📐 Checking Region Conditioning (Parallel)...")
    regions = get_regions_at_layer(tree, layer_idx)
    thin_count = Threads.Atomic{Int}(0)
    
    Threads.@threads for r in regions
        A_full, b_full = get_path_inequalities(r)
        _, r_val = get_chebyshev_center_radius(A_full, b_full)
        
        if r_val < min_radius
            Threads.atomic_add!(thin_count, 1)
        end
    end
    
    tc = thin_count[]
    if tc == 0
        println("✅ All regions are well-conditioned.")
        return true
    else
        println("⚠️ Found $tc thin regions (radius < $min_radius).")
        return false
    end
end

function get_chebyshev_center_radius(A, b; limit=1e5)
    m, n = size(A)
    if m == 0; return zeros(n), Inf; end
    
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
    
    target_layer = isnothing(layer_idx) ? tree.L : layer_idx
    println("ℹ️  Targeting Layer: $target_layer")

    results = Dict{String, Bool}()
    
    t_start = time()
    results["Hierarchy"] = verify_tree_hierarchy_robust(tree; bound=bound)
    results["Conditioning"] = check_region_conditioning(tree, target_layer)
    results["Completeness"] = verify_partition_completeness_monte_carlo(tree, target_layer, bound)
    
    dt = time() - t_start
    
    println("\n" * "="^60)
    println("📝 VERIFICATION SUMMARY (Time: $(round(dt, digits=2))s)")
    println("="^60)
    
    all_passed = true
    for (test, passed) in results
        status = passed ? "✅ PASS" : "❌ FAIL"
        @printf("%-15s : %s\n", test, status)
        if !passed; all_passed = false; end
    end
    
    println("-"^60)
    return all_passed
end

end # module