
using Random
using LinearAlgebra


mutable struct Trainer_CL
    input::Vector{Tuple{Int,Float64,Float64}}   # (edge, input strain, stiffness)
    output::Vector{Tuple{Int,Float64,Float64,Float64}} # (edge, target strain, stiffness,current accumulated free strain)
    edgetype::Vector{Int}    # 0: free, 1: input, 2: output
    net_f::ENM
    net_c::ENM
    G::Vector{Float64}
    
end

function Trainer_CL(net::ENM,
                   input::Vector{Tuple{Int,Float64,Float64}},
                   output::Vector{Tuple{Int,Float64,Float64}})

    edgetype = zeros(Int, net.ne)
    net_f = net
    net_c = deepcopy(net)

    # modify training masks + rest lengths + stiffnesses
    for (edge, strain, stiff) in input
        edgetype[edge] = 1
        net_f.l0[edge] *= (1 + strain)
        net_c.l0[edge] *= (1 + strain)
        net_f.k[edge] = stiff
        net_c.k[edge] = stiff
    end

    for (edge, _, _) in output
        edgetype[edge] = 2
        
    end
    #memory for learning
    G = zeros(net.ne)
    output = [(edge, target_strain, stiff, 0.0) for (edge, target_strain, stiff) in output]

    return Trainer_CL(input, output, edgetype, net_f, net_c, G)
end



function clamp_eta!(tr::Trainer_CL,step_md; eta)
    
    for (edge, strain_t, stiff,current_strain) in tr.output
        strain_f = current_strain/step_md
        strain_c = strain_f + (strain_t - strain_f)*eta
        put_stain!(tr.net_c, edge, strain_c; k=stiff)
    end
end

function update_info!(tr::Trainer_CL)
    
    @inbounds for (ide, (u,v)) in enumerate(tr.net_f.edges)
        if tr.edgetype[ide] == 0
            lf = norm(tr.net_f.pts[v,:] .- tr.net_f.pts[u,:])
            l0f = tr.net_f.l0[ide]

            lc = norm(tr.net_c.pts[v,:] .- tr.net_c.pts[u,:])
            l0c = tr.net_c.l0[ide]
            tr.G[ide] +=  (lc - l0c)^2 - (lf - l0f)^2    
        end
        
    end

    # update current free strains on output edges
    for (edge,_,_,current_strain) in tr.output
        current_strain += cal_strain(tr.net_f, edge)
    end

end

function learn_k!(tr::Trainer_CL, alpha,vmin=1e-3, vmax=1e2)
    
    for (ide, g) in enumerate(tr.G)
        if tr.edgetype[ide] == 0
            tr.net_f.k[ide] -= alpha * g
            tr.net_c.k[ide] -= alpha * g
            if tr.net_f.k[ide] < vmin
                tr.net_f.k[ide] = vmin
                tr.net_c.k[ide] = vmin
            elseif tr.net_f.k[ide] > vmax
                tr.net_f.k[ide] = vmax
                tr.net_c.k[ide] = vmax
            end
        end
    end
end 

function step!(tr::Trainer_CL,T; eta=1.0, alpha=1.0, step_md=10)

    
    clamp_eta!(tr,step_md; eta=eta)
    tr.output = [(edge, target_strain, stiff, 0.0) for (edge, target_strain, stiff, _) in tr.output]
    fill!(tr.G, 0.0)
    for _ in 1:step_md
        run_md!(tr.net_f,T)
        run_md!(tr.net_c,T)
        update_info!(tr)
    end
    # stiffness update
    learn_k!(tr, alpha)

   
    return [v[4] / step_md for v in tr.output]  # return current free strains on output edges 
end

