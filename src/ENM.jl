
using LinearAlgebra, Random
using Plots

struct ENM
    n::Int
    ne::Int
    pts0::Matrix{Float64}     # n × 3
    pts::Matrix{Float64}     # n × 3
    vel::Matrix{Float64}     # n × 3
    edges::Vector{Tuple{Int,Int}}
    k::Vector{Float64}
    l0::Vector{Float64}
    m::Float64 #assuming uniform mass
    
end

function load_graph(filename)
 ####---------------------
 # load graph from a text file
 # format:
 # n
 # x1 y1 z1
 # x2 y2 z2
 # ...
 # xn yn zn
 # ne
 # node1_1 node1_2 stiffness1 l0_1
 # node2_1 node2_2 stiffness2 l0_2
 # ...   
 ####---------------------
    lines = readlines(filename)
    idx = 1
    ### --- Read points ---
    n = parse(Int, lines[idx]); idx += 1
    pts = zeros(n, 3)
    for i in 1:n
        x, y, z = parse.(Float64, split(lines[idx]))
        pts[i, :] .= (x, y, z)
        idx += 1
    end

    ### --- Read edges ---
    ne = parse(Int, lines[idx]); idx += 1
    edges = Vector{Tuple{Int,Int}}(undef, ne)
    stiffness = Vector{Float64}(undef, ne)
    l0 = Vector{Float64}(undef, ne)

    for e in 1:ne
        cols = split(lines[idx])
        node1 = parse(Int, cols[1])
        node2 = parse(Int, cols[2])
        stiff = parse(Float64, cols[3])
        l0val = parse(Float64, cols[4])

        edges[e] = (node1+1, node2+1)
        stiffness[e] = stiff
        l0[e] = l0val

        idx += 1
    end

    return n, ne, pts, edges, stiffness, l0
end

function save_graph(filename, enm::ENM)
    open(filename, "w") do io
        println(io, enm.n)
        for i in 1:enm.n
            println(io, join(enm.pts[i, :], " "))
        end
        println(io, enm.ne)
        for e in 1:enm.ne
            u, v = enm.edges[e]
            println(io, "$(u-1) $(v-1) $(enm.k[e]) $(enm.l0[e])")
        end
    end
end



function ENM(filename; m=1.0, T0=0.0,seed=123)
    
    
    n, ne, pts, edges,kvec, lvec = load_graph(filename)
    pts0=deepcopy(pts)
    vel   = zeros(n, 3)
    

    # initial velocities
    Random.seed!(seed)
    for i in 1:n
        vel[i, :] = randn(3) * sqrt(T0/m)
    end

    return ENM(n, ne, pts0,pts, vel, edges, kvec, lvec,
               m)
end

function cal_degree(enm::ENM)
    deg = zeros(Int, enm.n)
    @inbounds for (u,v) in enm.edges
        deg[u] += 1
        deg[v] += 1
    end
    return deg
end


function cal_elastic_force!(force::AbstractMatrix{<:Real}, enm::ENM)
    
    fill!(force, 0.0)

    pts   = enm.pts
    edges = enm.edges
    k     = enm.k
    l0    = enm.l0

    @inbounds for i in 1:enm.ne
        u, v = edges[i]

        dx1 = pts[v,1] - pts[u,1]
        dx2 = pts[v,2] - pts[u,2]
        dx3 = pts[v,3] - pts[u,3]

        r2 = dx1*dx1 + dx2*dx2 + dx3*dx3
        if r2 == 0.0
            continue
        end
        r = sqrt(r2)

        # Hookean spring: f = k (r - l0) * (dx / r)
        scale = k[i] * (r - l0[i]) / r

        fx = scale * dx1
        fy = scale * dx2
        fz = scale * dx3

        force[u,1] += fx;  force[u,2] += fy;  force[u,3] += fz
        force[v,1] -= fx;  force[v,2] -= fy;  force[v,3] -= fz
    end

    return nothing
end


function cal_elastic_energy(enm::ENM)
    E = 0.0

    @inbounds for i in 1:enm.ne
        (u,v) = enm.edges[i]

        dx = enm.pts[v, :] .- enm.pts[u, :]
        dist = norm(dx)

        if dist == 0
            continue
        end

        delta = dist - enm.l0[i]
        E += 0.5 * enm.k[i] * delta^2
    end

    return E
end

function cal_kinetic_energy(enm::ENM)
    KE = 0.0
    @inbounds for i in 1:enm.n
        KE += 0.5 * enm.m * sum(enm.vel[i, :].^2)
    end
    return KE
end

function cal_strain(enm::ENM, edge::Int,l0::Float64= nothing)
    (u, v) = enm.edges[edge]
    dx = enm.pts[v, :] .- enm.pts[u, :]
    dist = norm(dx)
    l0 = l0 === nothing ? enm.l0[edge] : l0
    return (dist - l0) / l0
end

function put_strain!(enm::ENM, edge::Int, l0::Float64;k=100)
    enm.l0[edge] = l0
    enm.k[edge] = k
    return nothing
end

function reset_config!(enm::ENM)
    enm.pts .= enm.pts0
    fill!(enm.vel, 0.0)
    
    return nothing
end
#GJF method
function run_md!(
    enm::ENM, T;
    steps::Int=1,
    dt::Float64=0.005,
    tau::Float64=1.0,
    rng::AbstractRNG = Random.default_rng())
    
    m = enm.m
    α = m / tau
    c = α * dt / (2m)          # = dt/(2τ)
    a = (1 - c) / (1 + c)
    b = 1 / (1 + c)

    βscale = sqrt(2 * α * T * dt)   # kB=1

    β    = similar(enm.vel)
    f_old = similar(enm.vel)
    f_new = similar(enm.vel)

    # f_old = f(r^0)
    cal_elastic_force!(f_old, enm)
    

    @inbounds for _ in 1:steps
        # draw β^{n+1}
        randn!(rng, β)
        @. β *= βscale

        # r^{n+1}
        @. enm.pts += b * (dt * enm.vel + (dt^2/(2m)) * f_old + (dt/(2m)) * β)

        # f_new = f(r^{n+1})
        cal_elastic_force!(f_new, enm)

        # v^{n+1}
        @. enm.vel = a * enm.vel + (dt/(2m)) * (a * f_old + f_new) + (b/m) * β

        # next step: f_old ← f_new (swap buffers, no copy)
        f_old, f_new = f_new, f_old
    end
    return nothing
end


function quench_fire!(enm::ENM; dt::Float64=0.005, max_steps::Int=100_000, force_tol::Float64=1e-6)

    # FIRE parameters
    α     = 0.1; finc  = 1.1
    fdec  = 0.5; αdec  = 0.99
    dtmax = 10dt; Nmin  = 5
    Npos  = 0

    # unpack
    pts  = enm.pts; vel  = enm.vel
    F    = similar(enm.vel); m    = enm.m
    n    = enm.n

    for step in 1:max_steps

        cal_elastic_force!(F, enm)

        maxF = maximum(abs.(F))
        if maxF < force_tol
            return (:converged, step, maxF)
        end

        
        # Velocity update (Euler)
        
        @inbounds for i in 1:n
            vel[i,1] += dt * F[i,1] / m
            vel[i,2] += dt * F[i,2] / m
            vel[i,3] += dt * F[i,3] / m
        end

        # Fire Power
        P = sum(vel .* F)

        if P > 0
            Npos += 1
            if Npos > Nmin
                dt = min(dt * finc, dtmax)
                α *= αdec
            end
        else
            # negative power → reset
            Npos = 0
            dt *= fdec
            α = 0.1
            fill!(vel, 0.0)
        end

       
        # Velocity mixing
        vnorm = sqrt(sum(vel.^2))
        fnorm = sqrt(sum(F.^2))

        if vnorm > 0 && fnorm > 0
            mix = α * fnorm / vnorm
            vel .= (1 - α) .* vel .+ mix .* F
        end


        # Position update
        pts .+= dt .* vel
    end

    return (:max_steps_reached, max_steps, maximum(abs.(F)))
end

 
function cal_elastic_jacobian(enm::ENM)
    J = zeros(3*enm.n, 3*enm.n)

    @inbounds for i in 1:enm.ne
        (u,v) = enm.edges[i]

        dx = enm.pts[v,:] .- enm.pts[u,:]
        dist = norm(dx)
        dist == 0 && continue

        k  = enm.k[i]
        l0 = enm.l0[i]

        term1 = k*(1 - l0/dist)
        term2 = k*l0/(dist^3)

        for a in 1:3
            for b in 1:3
                val = term1*(a==b) + term2*dx[a]*dx[b]
                J[3u-2+a, 3u-2+b] += val
                J[3v-2+a, 3v-2+b] += val
                J[3u-2+a, 3v-2+b] -= val
                J[3v-2+a, 3u-2+b] -= val
            end
        end
    end
    return J
end

function cal_modes(enm::ENM)
    J = cal_elastic_jacobian(enm)
    eigen(J)
end


function plot_net(enm::ENM; camera=(30, 30), color=:nothing)
    x = enm.pts[:, 1]
    y = enm.pts[:, 2]
    z = enm.pts[:, 3]

    plt = plot3d(legend=false, camera=camera)

    scatter3d!(plt, x, y, z;
               markersize=5,
               markercolor=:blue,
               label="Nodes")

    # prepare edge colors when requested
    edge_colors = nothing
    if color == "k" || color == :k
        kvals = enm.k
        kmin = minimum(kvals)
        kmax = maximum(kvals)
        rng = kmax > kmin ? (kmax - kmin) : 1.0
        cmap = cgrad(:viridis)
        edge_colors = [cmap((kv - kmin) / rng) for kv in kvals]
    end

    for i in 1:enm.ne
        u, v = enm.edges[i]
        lc = edge_colors === nothing ? :black : edge_colors[i]
        plot3d!(plt,
                [x[u], x[v]],
                [y[u], y[v]],
                [z[u], z[v]];
                linecolor=lc,
                label="")
    end

    return plt
end

