# Verification logic

# ==============================================================================
# 1. VOLUME CALCULATION & BOUNDING
# ==============================================================================

function get_region_volume(region::Region; bound::Union{Float64, Nothing}=nothing)
    # 1. Get Region Constraints
    A, b = get_path_inequalities(region)
    dim = size(A, 2)
    
    # 2. Add Bounding Box Constraints if requested
    if !isnothing(bound)
        I_mat = Matrix{Float64}(I, dim, dim)
        A_box = vcat(I_mat, -I_mat)
        b_box = fill(bound, 2 * dim)
        
        A = vcat(A, A_box)
        b = vcat(b, b_box)
    end

    # 3. Construct Polyhedron
    h = hrep(A, b)
    poly = polyhedron(h, CDDLib.Library())

    # 4. Check Unboundedness
    # Fix: 'isbounded' is missing in some Polyhedra versions/namespaces.
    # We check if the V-representation has any rays instead.
    if isnothing(bound)
        try
            # Convert to V-representation (vertices + rays)
            vr = vrep(poly)
            # If there are any rays, the volume is infinite
            if !isempty(rays(vr))
                return Inf
            end
        catch
            # If conversion fails, assume complex/degenerate or handle downstream
        end
    end

    # 5. Compute Volume
    try
        removevredundancy!(poly)
        if isempty(poly)
            return 0.0
        end
        return volume(poly)
    catch e
        return 0.0
    end
end

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
    p = Progress(length(regions); desc="Computing Volumes: ")
    
    for r in regions
        vol = get_region_volume(r, bound=bound)
        total_region_vol += vol
        next!(p)
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
