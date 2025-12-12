
using Random
using LinearAlgebra

const INPUT = 1
const OUTPUT = 2
const NORMAL = 0


mutable struct Trainer_CL
    input::Vector{Tuple{Int,Float64,Float64,Float64}}   # (edge, input strain, stiffness, l0)
    output::Vector{Tuple{Int,Float64,Float64,Float64}} # (edge, target strain, stiffness,l0)
    trainable_edges::Vector{Int}    # 0: free, 1: input, 2: output
    net_f::ENM
    net_c::ENM
end

function Trainer_CL(net::ENM,
                   input::Vector{Tuple{Int,Float64,Float64}},
                   output::Vector{Tuple{Int,Float64,Float64}})

    edgetype = zeros(Int, net.ne)
    net_f = deepcopy(net)
    net_c = deepcopy(net)

    # modify training masks + rest lengths + stiffnesses
    input_construct = [(edge, strain, stiff, net.l0[edge]) for (edge, strain, stiff) in input]
    for (edge, strain, stiff, l0) in input_construct
        edgetype[edge] = INPUT
        put_strain!(net_f, edge, l0*(1+strain); k=stiff)
        put_strain!(net_c, edge, l0*(1+strain); k=stiff)
    end
    
    output_construct = [(edge, strain, stiff, net.l0[edge]) for (edge, strain, stiff) in output]
    for (edge, strain, stiff, l0) in output_construct
        edgetype[edge] = OUTPUT
        net_f.k[edge] = 0.0   
        net_c.k[edge] = 0.0
    end

    trainable_edges = findall(==(0), tr.edge_type)

    return Trainer_CL(input_construct, output_construct, trainable_edges, net_f, net_c)
end



function clamp_eta!(tr::Trainer_CL,strain_f::Vector{Float64},eta)
    
    for (ido,(edge, strain_t, stiff, l0)) in enumerate(tr.output)
        strain_c = strain_f[ido] + (strain_t - strain_f[ido])*eta
        put_strain!(tr.net_c, edge, l0*(1+strain_c); k=stiff)
    end
end


function update_k!(tr::Trainer_CL, grad::Vector{Float64}, alpha, vmin=1e-3, vmax=200.0)
    kf = tr.net_f.k
    kc = tr.net_c.k
    idxs = tr.trainable_edges   # indices where edge_type == 0

    @inbounds @simd for j in eachindex(idxs)
        i = idxs[j]

        newk = kf[i] - alpha * grad[i]
        newk = clamp(newk, vmin, vmax)

        kf[i] = newk
        kc[i] = newk   
    end

    return nothing
end


function update_Gradient!(tr::Trainer_CL, grad::Vector{Float64})
    pts_f  = tr.net_f.pts    # size (N, 3)
    pts_c  = tr.net_c.pts    # size (N, 3)
    l0f    = tr.net_f.l0
    l0c    = tr.net_c.l0
    edges  = tr.net_f.edges  
        

    @inbounds @simd for ide in eachindex(l0f)
        
        u = edges[ide, 1]
        v = edges[ide, 2]
    

    
        dx_f = pts_f[v, 1] - pts_f[u, 1]
        dy_f = pts_f[v, 2] - pts_f[u, 2]
        dz_f = pts_f[v, 3] - pts_f[u, 3]
        lf   = sqrt(dx_f*dx_f + dy_f*dy_f + dz_f*dz_f)

    
        dx_c = pts_c[v, 1] - pts_c[u, 1]
        dy_c = pts_c[v, 2] - pts_c[u, 2]
        dz_c = pts_c[v, 3] - pts_c[u, 3]
        lc   = sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c)

        δc = lc - l0c[ide]
        δf = lf - l0f[ide]
        grad[ide] += δc*δc - δf*δf
    end
end



function step!(tr::Trainer_CL,T;sf_old= nothing, eta=1.0, alpha=1.0, step_md=10)
    #learning information reset
    Gradient=zeros(length(tr.net_f.edges))
    sf_new = zeros(length(tr.output))

    if sf_old != nothing
        clamp_eta!(tr,sf_old,eta)
    else
        clamp_eta!(tr,zeros(length(tr.output)), eta)
    end
    

    
    
    for _ in 1:step_md
        run_md!(tr.net_f,T)
        run_md!(tr.net_c,T)
        # update gradient
        update_Gradient!(tr, Gradient)
        #update current free strains
        for (ido,(edge,_,_,l0)) in enumerate(tr.output)
            sf_new[ido] += cal_strain(tr.net_f, edge, l0)
        end
    end

    Gradient=Gradient ./ step_md
    sf_new .= sf_new ./ step_md
    
    # stiffness update
    learn_k!(tr, Gradient, alpha)
   
    return sf_new # return current free strains on output edges 
end

