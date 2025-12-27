
using LinearAlgebra, Random
using Plots

struct ENM
    dim::Int                    #adding dim
    n::Int                   # number of nodes
    pts0::Matrix{Float64}     # n × 3
    pts::Matrix{Float64}     # n × 3
    vel::Matrix{Float64}     # n × 3
    edges::Vector{Tuple{Int,Int}} #ne edges
    k::Vector{Float64}
    l0::Vector{Float64}
    m::Float64 #assuming uniform mass

end

function load_graph(filename)
 ####---------------------
 # load graph from a text file
 # format:
 #dim
 # n
 # x1 y1 (z1)
 # x2 y2 (z2)
 # ...
 # xn yn (zn)
 # ne
 # node1_1 node1_2 stiffness1 l0_1
 # node2_1 node2_2 stiffness2 l0_2
 # ...   
 ####---------------------
    lines = readlines(filename)
    idx = 1
    ### --- Read points ---
    dim = parse(Int, lines[idx]); idx += 1
    n = parse(Int, lines[idx]); idx += 1
    pts = zeros(n, dim)
    for i in 1:n
        coords = parse.(Float64, split(lines[idx]))
        pts[i, :] .= coords
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

    return dim,n, pts, edges, stiffness, l0
end

function save_enm(filename, enm::ENM)
    open(filename, "w") do io
        println(io, enm.dim)  
        println(io, enm.n)
        for i in 1:enm.n
            println(io, join(enm.pts[i, :], " "))
        end
        println(io, length(enm.edges))
        for e in 1:length(enm.edges)
            u, v = enm.edges[e]
            println(io, "$(u-1) $(v-1) $(enm.k[e]) $(enm.l0[e])")
        end
    end
end



function ENM(filename; m=1.0, T0=0.0,seed=123)
    
    
    dim,n, pts, edges,kvec, lvec = load_graph(filename)
    pts0=deepcopy(pts)
    vel   = zeros(n, dim)
    

    # initial velocities
    Random.seed!(seed)
    for i in 1:n
        vel[i, :] = randn(dim) * sqrt(T0/m)
    end

    return ENM(dim,n, pts0,pts, vel, edges, kvec, lvec,
               m)
end

function add_edge!(enm::ENM, nodei::Int, nodej::Int; k::Float64=0.0)
    #adding edge without strain ( l0 set to current distance)
    if nodei < 1 || nodei > enm.n || nodej < 1 || nodej > enm.n
        error("Node indices out of bounds.")
    end
    if (nodei,nodej) in enm.edges || (nodej,nodei) in enm.edges
        error("Edge already exists between node $nodei and node $nodej.")
    end

    push!(enm.edges, (nodei, nodej))
    l0_norm=norm(enm.pts[nodei,:]-enm.pts[nodej,:])
    push!(enm.l0, l0_norm)
    push!(enm.k, k)
    return length(enm.edges)
end

function cal_degree(enm::ENM)
    deg = zeros(Int, enm.n)
    @inbounds for (u,v) in enm.edges
        deg[u] += 1
        deg[v] += 1
    end
    return deg
end


function cal_elastic_force!(force::AbstractMatrix{T}, enm::ENM) where {T<:Real}
    fill!(force, zero(T))

    pts   = enm.pts
    edges = enm.edges
    k     = enm.k
    l0    = enm.l0
    ne= length(edges)
    dim = enm.dim
    @assert dim == 2 || dim == 3  #dim is either 2 or 3

    if dim == 2
        @inbounds for i in 1:ne
            u, v = edges[i]

            dx1 = pts[v,1] - pts[u,1]
            dx2 = pts[v,2] - pts[u,2]

            r2 = dx1*dx1 + dx2*dx2
            r2 == 0 && continue

            r = sqrt(r2)
            s = k[i] * (r - l0[i]) / r

            f1 = s*dx1
            f2 = s*dx2

            force[u,1] += f1; force[v,1] -= f1
            force[u,2] += f2; force[v,2] -= f2
        end

    else # dim == 3
        @inbounds for i in 1:ne
            u, v = edges[i]

            dx1 = pts[v,1] - pts[u,1]
            dx2 = pts[v,2] - pts[u,2]
            dx3 = pts[v,3] - pts[u,3]

            r2 = dx1*dx1 + dx2*dx2 + dx3*dx3
            r2 == 0 && continue

            r = sqrt(r2)
            s = k[i] * (r - l0[i]) / r

            f1 = s*dx1
            f2 = s*dx2
            f3 = s*dx3

            force[u,1] += f1; force[v,1] -= f1
            force[u,2] += f2; force[v,2] -= f2
            force[u,3] += f3; force[v,3] -= f3
        end
    end

    return nothing
end



function cal_elastic_energy(enm::ENM)
    E = 0.0
    ne    = length(enm.edges)
    @inbounds for i in 1:ne
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

function cal_strain(enm::ENM, edge::Int)
    (nodei, nodej) = enm.edges[edge]
     # calculate strain of a specific edge
    dist = norm(enm.pts[nodej, :] .- enm.pts[nodei, :])
    l0 = norm(enm.pts0[nodej, :] .- enm.pts0[nodei, :])
   
    return (dist - l0) / l0
end

#put strain on an adge
function put_strain!(enm::ENM, edge::Int,strain::Float64; k=100)
    (u,v)= enm.edges[edge]
    l0 = norm(enm.pts0[v, :] .- enm.pts0[u, :])
    enm.l0[edge] = l0 * (1 + strain)
    enm.k[edge] = k
    return nothing
end

function reset_config!(enm::ENM)
    enm.pts .= enm.pts0
    fill!(enm.vel, 0.0)
    
    return nothing
end


#  ---------dynamics ---------
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
    α0    = 0.1
    α     = α0
    finc  = 1.1
    fdec  = 0.5
    αdec  = 0.99
    dtmax = 10*dt
    Nmin  = 5
    Npos  = 0

    # unpack
    pts  = enm.pts
    vel  = enm.vel
    F    = similar(vel)
    m    = enm.m
    n    = enm.n
    dim  = enm.dim
    @assert dim == 2 || dim == 3

    @inbounds for step in 1:max_steps
        cal_elastic_force!(F, enm)

        # maxF = maximum(abs.(F)) but without allocation
        maxF = 0.0
        if dim == 2
            for i in 1:n
                f1 = F[i,1]; f2 = F[i,2]
                af1 = abs(f1); af2 = abs(f2)
                maxF = max(maxF, af1, af2)
            end
        else
            for i in 1:n
                f1 = F[i,1]; f2 = F[i,2]; f3 = F[i,3]
                af1 = abs(f1); af2 = abs(f2); af3 = abs(f3)
                maxF = max(maxF, af1, af2, af3)
            end
        end

        if maxF < force_tol
            return (:converged, step, maxF)
        end

        # Euler velocity update + compute power and norms in the same pass
        P     = 0.0
        v2sum = 0.0
        f2sum = 0.0

        if dim == 2
            for i in 1:n
                f1 = F[i,1]; f2 = F[i,2]

                v1 = vel[i,1] + dt * f1 / m
                v2 = vel[i,2] + dt * f2 / m
                vel[i,1] = v1
                vel[i,2] = v2

                P     += v1*f1 + v2*f2
                v2sum += v1*v1 + v2*v2
                f2sum += f1*f1 + f2*f2
            end
        else
            for i in 1:n
                f1 = F[i,1]; f2 = F[i,2]; f3 = F[i,3]

                v1 = vel[i,1] + dt * f1 / m
                v2 = vel[i,2] + dt * f2 / m
                v3 = vel[i,3] + dt * f3 / m
                vel[i,1] = v1
                vel[i,2] = v2
                vel[i,3] = v3

                P     += v1*f1 + v2*f2 + v3*f3
                v2sum += v1*v1 + v2*v2 + v3*v3
                f2sum += f1*f1 + f2*f2 + f3*f3
            end
        end

        # FIRE timestep / alpha adaptation
        if P > 0
            Npos += 1
            if Npos > Nmin
                dt = min(dt * finc, dtmax)
                α *= αdec
            end
        else
            Npos = 0
            dt *= fdec
            α = α0
            fill!(vel, 0.0)
            # (Optional but common: continue to next step after reset)
        end

        # Velocity mixing: v <- (1-α)v + α |v|/|F| F
        if v2sum > 0 && f2sum > 0
            vnorm = sqrt(v2sum)
            fnorm = sqrt(f2sum)
            s = α * vnorm / fnorm   # standard FIRE form

            if dim == 2
                for i in 1:n
                    vel[i,1] = (1-α)*vel[i,1] + s*F[i,1]
                    vel[i,2] = (1-α)*vel[i,2] + s*F[i,2]
                end
            else
                for i in 1:n
                    vel[i,1] = (1-α)*vel[i,1] + s*F[i,1]
                    vel[i,2] = (1-α)*vel[i,2] + s*F[i,2]
                    vel[i,3] = (1-α)*vel[i,3] + s*F[i,3]
                end
            end
        end

        # Position update: r <- r + dt v
        if dim == 2
            for i in 1:n
                pts[i,1] += dt * vel[i,1]
                pts[i,2] += dt * vel[i,2]
            end
        else
            for i in 1:n
                pts[i,1] += dt * vel[i,1]
                pts[i,2] += dt * vel[i,2]
                pts[i,3] += dt * vel[i,3]
            end
        end
    end

    # if we exit loop, F is from last step; maxF recompute cheaply:
    cal_elastic_force!(F, enm)
    maxF = 0.0
    @inbounds for x in F
        ax = abs(x)
        maxF = ax > maxF ? ax : maxF
    end

    return (:max_steps_reached, max_steps, maxF)
end


#--------- mode analysis --------- 

function cal_elastic_jacobian(enm::ENM)
    dim = enm.dim
    @assert dim == 2 || dim == 3

    T = eltype(enm.pts)
    J = zeros(T, dim*enm.n, dim*enm.n)

    pts   = enm.pts
    edges = enm.edges
    k     = enm.k
    l0    = enm.l0
    ne    = length(edges)
    @inbounds for i in 1:ne
        u, v = edges[i]

        # dx and dist (no allocation)
        dist2 = zero(T)
        dx = ntuple(d -> pts[v,d] - pts[u,d], dim)
        @inbounds for d in 1:dim
            dist2 += dx[d]*dx[d]
        end
        dist2 == 0 && continue
        dist = sqrt(dist2)

        ki  = k[i]
        l0i = l0[i]

        term1 = ki * (1 - l0i/dist)
        term2 = ki * l0i / (dist^3)

        bu = (u-1)*dim
        bv = (v-1)*dim

        @inbounds for a in 1:dim, b in 1:dim
            val = term1*(a==b) + term2*dx[a]*dx[b]
            J[bu+a, bu+b] += val
            J[bv+a, bv+b] += val
            J[bu+a, bv+b] -= val
            J[bv+a, bu+b] -= val
        end
    end

    return J
end


function cal_modes(enm::ENM)
    J = cal_elastic_jacobian(enm)
    eigen(J)
end

#--------- plotting functions ---------


function plot_net(enm::ENM;
    input:: Vector{Int}= nothing,#edge
    output:: Vector{Int}= nothing,#edge
    camera = (30, 30),
    color:: Vector{Float64}=nothing)
    if color !== nothing
        if length(color) != length(enm.edges)
            error("Length of color vector must match number of edges.")
        end
    end
    if enm.dim == 2
        return plot_net_2d(enm; input=input, output=output, color=color)
    elseif enm.dim == 3
        return plot_net_3d(enm; input=input, output=output, camera=camera, color=color)
    else
        error("Unsupported dimension: $(enm.dim). Only 2D and 3D are supported.")
    end
end


using CairoMakie 

function plot_net_2d(
    enm::ENM;
    input::Union{Nothing,Vector{Int}} = nothing,
    output::Union{Nothing,Vector{Int}} = nothing,
    color = nothing
 )
    pts   = enm.pts
    edges = enm.edges
    ne    = length(edges)

    @assert size(pts, 2) == 2 "enm.pts must be N×2"
    N = size(pts, 1)

    input_set  = input  === nothing ? Set{Int}() : Set(input)
    output_set = output === nothing ? Set{Int}() : Set(output)

    # normal edges are those NOT in input or output
    normal_idxs = Vector{Int}()
    sizehint!(normal_idxs, ne)
    for i in 1:ne
        if !(i in input_set) && !(i in output_set)
            push!(normal_idxs, i)
        end
    end

    # --- Build line collections as vectors of Point2f: pairs are segments ---
    # normal segments
    normal_seg = Point2f[]
    sizehint!(normal_seg, 2 * length(normal_idxs))
    for idx in normal_idxs
        u, v = edges[idx]
        @assert 1 ≤ u ≤ N && 1 ≤ v ≤ N "Edge has node index out of bounds: ($u,$v)"
        push!(normal_seg, Point2f(pts[u, 1], pts[u, 2]))
        push!(normal_seg, Point2f(pts[v, 1], pts[v, 2]))
    end

    # input segments
    input_seg = Point2f[]
    if !isempty(input_set)
        sizehint!(input_seg, 2 * length(input_set))
        for idx in input_set
            u, v = edges[idx]
            push!(input_seg, Point2f(pts[u, 1], pts[u, 2]))
            push!(input_seg, Point2f(pts[v, 1], pts[v, 2]))
        end
    end

    # output segments
    output_seg = Point2f[]
    if !isempty(output_set)
        sizehint!(output_seg, 2 * length(output_set))
        for idx in output_set
            u, v = edges[idx]
            push!(output_seg, Point2f(pts[u, 1], pts[u, 2]))
            push!(output_seg, Point2f(pts[v, 1], pts[v, 2]))
        end
    end

    # --- Figure / Axis ---
    fig = Figure(resolution = (500, 500))
    ax  = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y")

    # --- Normal edges (colormapped or black) ---
    normal_plot = nothing

    if color === nothing
        if !isempty(normal_seg)
            linesegments!(ax, normal_seg; color = :grey, linewidth = 1.5)
        end
    else
        # values used for coloring only normal edges
        vals = if (color == :k || color == "k")
            @assert hasproperty(enm, :k) "color=:k requires enm.k"
            enm.k[normal_idxs]
        elseif color isa AbstractVector{<:Real}
            @assert length(color) == ne "If `color` is a vector, it must have length == number of edges"
            color[normal_idxs]
        else
            error("Unsupported `color`. Use nothing, :k (or \"k\"), or a length-ne vector of reals.")
        end

        if isempty(vals) || isempty(normal_seg)
            # nothing normal to color; skip colorbar
        else
            cmin = minimum(vals)
            cmax = maximum(vals)
            rng  = (cmax > cmin) ? (cmax - cmin) : 1.0
            colorrange = (cmin, cmax)

            # `linesegments` accepts one scalar per segment (i.e., per edge)
            # Here: number of segments == length(normal_idxs)
            normal_plot = linesegments!(
                ax, normal_seg;
                color      = vals,
                colormap   = :viridis,
                colorrange = colorrange,
                linewidth  = 1.8
            )

            # colorbar reflects ONLY normal edges
            Colorbar(fig[1, 2], normal_plot; label = "edge color (normal edges)")
        end
    end

    # --- Special edges (dashed, fixed colors; excluded from colorbar by construction) ---
    if !isempty(input_seg)
        linesegments!(ax, input_seg; color = :darkblue, linestyle = :dash, linewidth = 2.4)
    end
    if !isempty(output_seg)
        linesegments!(ax, output_seg; color = :darkred, linestyle = :dash, linewidth = 2.4)
    end

    # --- Nodes ---
    scatter!(ax, pts[:, 1], pts[:, 2]; color = :grey, markersize = 6)

    return fig
end

 
function plot_net_3d(
    enm::ENM;
    input::Union{Nothing,Vector{Int}} = nothing,
    output::Union{Nothing,Vector{Int}} = nothing,
    color = nothing
 )
    pts   = enm.pts
    edges = enm.edges
    ne    = length(edges)

    @assert size(pts, 2) == 3 "enm.pts must be N×3"
    N = size(pts, 1)

    input_set  = input  === nothing ? Set{Int}() : Set(input)
    output_set = output === nothing ? Set{Int}() : Set(output)

    # normal edges: not special
    normal_idxs = Vector{Int}()
    sizehint!(normal_idxs, ne)
    for i in 1:ne
        if !(i in input_set) && !(i in output_set)
            push!(normal_idxs, i)
        end
    end

    # --- Build line collections (pairs of Point3f are segments) ---
    normal_seg = Point3f[]
    sizehint!(normal_seg, 2 * length(normal_idxs))
    for idx in normal_idxs
        u, v = edges[idx]
        @assert 1 ≤ u ≤ N && 1 ≤ v ≤ N "Edge has node index out of bounds: ($u,$v)"
        push!(normal_seg, Point3f(pts[u, 1], pts[u, 2], pts[u, 3]))
        push!(normal_seg, Point3f(pts[v, 1], pts[v, 2], pts[v, 3]))
    end

    input_seg = Point3f[]
    if !isempty(input_set)
        sizehint!(input_seg, 2 * length(input_set))
        for idx in input_set
            u, v = edges[idx]
            push!(input_seg, Point3f(pts[u, 1], pts[u, 2], pts[u, 3]))
            push!(input_seg, Point3f(pts[v, 1], pts[v, 2], pts[v, 3]))
        end
    end

    output_seg = Point3f[]
    if !isempty(output_set)
        sizehint!(output_seg, 2 * length(output_set))
        for idx in output_set
            u, v = edges[idx]
            push!(output_seg, Point3f(pts[u, 1], pts[u, 2], pts[u, 3]))
            push!(output_seg, Point3f(pts[v, 1], pts[v, 2], pts[v, 3]))
        end
    end

    # --- Figure / LScene (3D) ---
    fig = Figure(resolution = (950, 750))
    ax3 = LScene(fig[1, 1], scenekw = (show_axis = true,))

    # --- Normal edges ---
    normal_plot = nothing
    if color === nothing
        if !isempty(normal_seg)
            linesegments!(ax3, normal_seg; color = :black, linewidth = 1.4)
        end
    else
        vals = if (color == :k || color == "k")
            @assert hasproperty(enm, :k) "color=:k requires enm.k"
            enm.k[normal_idxs]
        elseif color isa AbstractVector{<:Real}
            @assert length(color) == ne "If `color` is a vector, it must have length == number of edges"
            color[normal_idxs]
        else
            error("Unsupported `color`. Use nothing, :k (or \"k\"), or a length-ne vector of reals.")
        end

        if !isempty(vals) && !isempty(normal_seg)
            cmin = minimum(vals)
            cmax = maximum(vals)
            colorrange = (cmin, cmax)

            normal_plot = linesegments!(
                ax3, normal_seg;
                color      = vals,          # one value per edge segment
                colormap   = :viridis,
                colorrange = colorrange,
                linewidth  = 1.6
            )

            Colorbar(fig[1, 2], normal_plot; label = "edge color (normal edges)")
        end
    end

    # --- Special edges (dashed, fixed colors; excluded from colorbar) ---
    if !isempty(input_seg)
        linesegments!(ax3, input_seg; color = :darkblue, linestyle = :dash, linewidth = 2.2)
    end
    if !isempty(output_seg)
        linesegments!(ax3, output_seg; color = :darkred, linestyle = :dash, linewidth = 2.2)
    end

    # --- Nodes ---
    scatter!(ax3, pts[:, 1], pts[:, 2], pts[:, 3]; color = :black, markersize = 8)

    return fig
end

