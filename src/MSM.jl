# to build the markov state model, recording trajectory data during MD,
# documenting the projections onto the softest modes
# then do TICA and clustering to build MSM

using LinearAlgebra
using Random
using Statistics
using JLD2
using Clustering

function rigid_align!(X::Matrix{Float64}, Xref::Matrix{Float64})
    # subtract centroids
    cX    = mean(X; dims=1)
    cXref = mean(Xref; dims=1)

    Xc    = X    .- cX
    Xrefc = Xref .- cXref

    # covariance
    H = Xc' * Xrefc
    U, _, Vt = svd(H)
    R = Vt' * U'

    # reflection fix
    if det(R) < 0
        Vt[end, :] .*= -1
        R = Vt' * U'
    end

    # apply rotation + translation
    X .= (Xc * R) .+ cXref
    return nothing
end

@inline vec_pts(pts::Matrix{Float64}) = reshape(permutedims(pts), :, 1)[:,1]

function mode_basis(enm::ENM; k::Int=10)
    #dim=2 or 3 only
    if enm.dim == 2
        skip = 3   # 2 translations + 1 rotation
    else
        skip = 6   # 3 translations + 3 rotations
    end

    # Jacobian evaluated at pts0 by construction
    E = cal_modes(enm, false)
    p = sortperm(E.values)          # ascending
    p = p[(skip+1):min(skip+k, end)]

    Φ = E.vectors[:, p]             # ndof × k
    λ = E.values[p]
    return Φ, λ
end

"""
    project_modes_rigid(enm, Φ)

Rigid-align current configuration to pts0, then project onto soft modes.
"""
function project_modes_rigid(enm::ENM, Φ::Matrix{Float64})
    # make a working copy (do NOT modify enm.pts permanently)
    X = copy(enm.pts)

    # rigid alignment
    rigid_align!(X, enm.pts0)

    # displacement vector
    x  = vec_pts(X)
    x0 = vec_pts(enm.pts0)

    return Φ' * (x .- x0)
end


function tica(X::AbstractMatrix;
              lag::Int=10,
              k::Union{Nothing,Int}=nothing,
              center::Bool=true,
              standardize::Bool=true,
              reg::Real=1e-8,
              symmetrize::Bool=true)


    T, d = size(X)
    @assert 1 ≤ lag < T "lag must satisfy 1 ≤ lag < number of frames"

    # Work in Float64 for stability
    Xf = Array{Float64}(X)

    # Preprocess: center + (optional) standardize
    μ = center ? vec(mean(Xf; dims=1)) : zeros(Float64, d)
    Xc = Xf .- μ'

    σ = ones(Float64, d)
    if standardize
        σ = vec(std(Xc; dims=1))
        σ .+= 1e-12                 # avoid division by zero
        Xc ./= σ'
    end

    # Build time-lagged pairs
    X0 = @view Xc[1:(T-lag), :]
    Xτ = @view Xc[(1+lag):T, :]
    n = size(X0, 1)

    # Covariances
    C0 = (X0' * X0) / n
    Cτ = (X0' * Xτ) / n

    if symmetrize
        Cτ = 0.5 * (Cτ + Cτ')
    end

    # Regularize C0 (helps when features are correlated / near-singular)
    C0r = C0 + reg * I

    # Solve generalized eigenproblem: Cτ v = λ C0 v
    F = eigen(Symmetric(Cτ), Symmetric(C0r))
    vals = F.values
    vecs = F.vectors

    # Sort by descending eigenvalue magnitude (usually descending value)
    perm = sortperm(vals; rev=true)
    vals = vals[perm]
    vecs = vecs[:, perm]

    kk = (k === nothing) ? d : min(k, d)
    W = vecs[:, 1:kk]
    eigvals = vals[1:kk]
    project = function (Xnew::AbstractMatrix)
        Xn = Array{Float64}(Xnew)
        @assert size(Xn, 2) == d "Xnew must have the same feature dimension d=$d"
        Xn = Xn .- μ'
        if standardize
            Xn ./= σ'
        end
        return Xn * W
    end

    return (W=W, eigvals=eigvals, mean=μ, scale=σ, lag=lag, project=project)
end
