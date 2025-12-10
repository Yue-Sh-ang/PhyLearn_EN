
using LinearAlgebra, Random
using Statistics
"Container for storing distinct minima."
struct Minimum
    energy::Float64
    pts::Matrix{Float64}   # n × 3
end

function kabsch(P::Matrix{Float64}, Q::Matrix{Float64})
    @assert size(P) == size(Q) "P and Q must have the same size"
    H = P' * Q                           # 3×3 covariance
    F = svd(H)
    U, V = F.U, F.V
    R = U * V'
    if det(R) < 0.0
        U[:, end] .= -U[:, end]         # enforce proper rotation (det = +1)
        R = U * V'
    end
    return Matrix{Float64}(R)
end

"using kabsch algorithm to canonicalize pts"
function canonicalize_pts(pts::Matrix{Float64}, ref_centered::Matrix{Float64})
    @assert size(pts) == size(ref_centered)

    com = vec(mean(pts, dims=1))          # 3-element vector
    pts_centered = pts .- com'            # n×3

    # optimal rotation to align with reference
    R = kabsch(pts_centered, ref_centered)  # 3×3
    pts_aligned = pts_centered * R          # n×3

    return pts_aligned
end

function hash_minimum_rigid(pts::Matrix{Float64}, ref_centered::Matrix{Float64}; digits::Int=4)
    canon = canonicalize_pts(pts, ref_centered)
    rounded = round.(canon; digits=digits)
    return hash(rounded)
end


function basin_hopping!(enm::ENM;
                        nsteps::Int=1000,
                        T_bh::Float64=1e-6,
                        disp_sigma::Float64=0.04,
                        force_tol::Float64=1e-12,
                        max_quench_steps::Int=100_000,
                        dt_quench::Float64=0.005,
                        seed::Union{Nothing,Int}=nothing)

    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)

    # --- reference for rigid-motion-invariant hashing ---
    ref_pts0 = copy(enm.pts0)
    ref_com  = vec(mean(ref_pts0, dims=1))
    ref_centered = ref_pts0 .- ref_com'

    # --- start from a local minimum ---
    fill!(enm.vel, 0.0)
    quench_fire!(enm; dt=dt_quench, max_steps=max_quench_steps, force_tol=force_tol)
    E_curr = cal_elastic_energy(enm)

    minima = Dict{UInt64, Minimum}()
    key0   = hash_minimum_rigid(enm.pts, ref_centered)
    minima[key0] = Minimum(E_curr, copy(enm.pts))

    pts_backup = copy(enm.pts)
    vel_backup = copy(enm.vel)
    n = enm.n

    for step in 1:nsteps
        pts_backup .= enm.pts
        vel_backup .= enm.vel

        # random kick
        @inbounds for i in 1:n, d in 1:3
            enm.pts[i,d] = pts_backup[i,d] + disp_sigma * randn(rng)
        end
        fill!(enm.vel, 0.0)

        # quench
        quench_fire!(enm; dt=dt_quench, max_steps=max_quench_steps, force_tol=force_tol)
        E_new = cal_elastic_energy(enm)

        # hash in canonical frame
        key_new = hash_minimum_rigid(enm.pts, ref_centered)
        if !haskey(minima, key_new)
            minima[key_new] = Minimum(E_new, copy(enm.pts))
        end

        # Metropolis acceptance at T_bh
        ΔE = E_new - E_curr
        accept = (ΔE ≤ 0.0) || (rand(rng) < exp(-ΔE / T_bh))

        if accept
            E_curr = E_new
        else
            enm.pts .= pts_backup
            enm.vel .= vel_backup
        end
    end

    return minima
end
