
using Random
using LinearAlgebra

const INPUT = 1
const OUTPUT = 2
const NORMAL = 0

#information container for training
mutable struct Trainer_CL
    input::Vector{Tuple{Int,Float64,Float64}}   # (edge, input strain, l0)
    output::Vector{Tuple{Int,Float64,Float64}} # (edge, target strain, l0)  
    trainable_edges::Vector{Int}  # indices of trainable edges
    net_f::ENM
    net_c::ENM
end

function Trainer_CL(net::ENM,#the network from allo
                   input::Vector{Tuple{Int,Int,Float64}} , # (nodei,nodej,input strain)
                   output::Vector{Tuple{Int,Int,Float64}}  # (nodei,nodej,target strain)
                   )
    # this function prepares the basic setting for training
    # denote input and output edges and denote their 
    ne = length(net.edges)
    edgetype = zeros(Int, ne)
    input_construct = Vector{Tuple{Int,Float64,Float64}}()
    output_construct=Vector{Tuple{Int,Float64,Float64}}()

    net_f = deepcopy(net)
    
 #  denote input nodes
    for (i,j,st) in input
        if (i,j) in net_f.edges || (j,i) in net_f.edges
            id_input=findfirst(x->x==(i,j),net_f.edges)
            edgetype[id_input] = INPUT
            net_f.k[id_input] = 0
        else
            id_input=add_edge!(net_f, i, j)    
            push!(edgetype, INPUT)
        end
        l0=norm(net_f.pts0[j,:] .- net_f.pts0[i,:])
        push!(input_construct, (id_input, st, l0))
    end

 #  denote output nodes
    for (i,j,st) in output
        if (i,j) in net_f.edges || (j,i) in net_f.edges
            id_output=findfirst(x->x==(i,j),net_f.edges)
            edgetype[id_output] = OUTPUT
            net_f.k[id_output] = 0
        else
            id_output=add_edge!(net_f, i, j)
            push!(edgetype, OUTPUT)
        end
        l0=norm(net_f.pts0[j,:] .- net_f.pts0[i,:])
        push!(output_construct, (id_output, st, l0))
    end

    net_c = deepcopy(net_f)
    trainable_edges = findall(==(NORMAL), edgetype)
    return Trainer_CL(input_construct, output_construct, trainable_edges, net_f, net_c)
end


# LEGO functions for training
function set_edge_k!(net::ENM,edge::Int,k::Float64)
    net.k[edge] = k
end

function set_edge_l0!(net::ENM,edge::Int,l0::Float64)
    net.l0[edge] = l0
end

function update_k!(tr::Trainer_CL, grad::Vector{Float64}, alpha, vmin=1e-3, vmax=200.0)
    kf = tr.net_f.k
    kc = tr.net_c.k
       # indices where edge_type == 0
    idxs = tr.trainable_edges
    @inbounds @simd for j in eachindex(idxs)
        i = idxs[j]

        newk = kf[i] - alpha * grad[i]
        newk = clamp(newk, vmin, vmax)

        kf[i] = newk
        kc[i] = newk   
    end

    return nothing
end


function update_grad!(tr::Trainer_CL, grad::Vector{Float64})
    # using classical CL rule to update gradient
    pts_f  = tr.net_f.pts
    pts_c  = tr.net_c.pts
    l0f    = tr.net_f.l0
    l0c    = tr.net_c.l0
    edges  = tr.net_f.edges
    dim    = tr.net_f.dim

    @assert dim == 2 || dim == 3

    if dim == 2
        @inbounds @simd for ide in eachindex(l0f)
            u, v = edges[ide]

            dx_f = pts_f[v,1] - pts_f[u,1]
            dy_f = pts_f[v,2] - pts_f[u,2]
            lf   = sqrt(dx_f*dx_f + dy_f*dy_f)

            dx_c = pts_c[v,1] - pts_c[u,1]
            dy_c = pts_c[v,2] - pts_c[u,2]
            lc   = sqrt(dx_c*dx_c + dy_c*dy_c)

            δc = lc - l0c[ide]
            δf = lf - l0f[ide]
            grad[ide] += δc*δc - δf*δf
        end
    else
        @inbounds @simd for ide in eachindex(l0f)
            u, v = edges[ide]

            dx_f = pts_f[v,1] - pts_f[u,1]
            dy_f = pts_f[v,2] - pts_f[u,2]
            dz_f = pts_f[v,3] - pts_f[u,3]
            lf   = sqrt(dx_f*dx_f + dy_f*dy_f + dz_f*dz_f)

            dx_c = pts_c[v,1] - pts_c[u,1]
            dy_c = pts_c[v,2] - pts_c[u,2]
            dz_c = pts_c[v,3] - pts_c[u,3]
            lc   = sqrt(dx_c*dx_c + dy_c*dy_c + dz_c*dz_c)

            δc = lc - l0c[ide]
            δf = lf - l0f[ide]
            grad[ide] += δc*δc - δf*δf
        end
    end

    return nothing
end



# # one exampe of training: eta always 1
# function step!(tr::Trainer_CL,T;sf_old= nothing, eta=1.0, alpha=1.0, step_md=10)
#     #learning information reset
#     Gradient=zeros(length(tr.net_f.edges))
#     sf_new = zeros(length(tr.output))
#     #clamped
#     set_edge_l0!(tr.net_c,tr.output[1][1], (1+tr.output[1][2])*tr.output[1][3]) # set the first output edge to zero stiffness
#     set_edge_k!(tr.net_c,tr.output[1][1], 100)
#     run_md!(tr.net_c,T,step=500) # equilibrate the clamped network
    
    
#     for _ in 1:step_md
#         run_md!(tr.net_f,T)
#         run_md!(tr.net_c,T)
#         # update gradient
#         update_Gradient!(tr, Gradient)
#         #update current free strains
#         for (ido,(edge,_,_,l0)) in enumerate(tr.output)
#             nodei, nodej = tr.net_f.edges[edge]
#             dist=norm(tr.net_f.pts[nodej, :] .- tr.net_f.pts[nodei, :])
#             sf_new[ido] += (dist - l0)/l0
#         end
#     end

#     Gradient=Gradient ./ step_md
#     sf_new .= sf_new ./ step_md
    
#     # stiffness update
#     learn_k!(tr, Gradient, alpha)
   
#     return sf_new # return current free strains on output edges 
# end


# function step!(tr::Trainer_CL,T;sf_old= nothing, eta=1.0, alpha=1.0, step_md=10)
#     #learning information reset
#     Gradient=zeros(length(tr.net_f.edges))
#     sf_new = zeros(length(tr.output))

#     if sf_old != nothing
#         clamp_eta!(tr,sf_old,eta)
#     else
#         clamp_eta!(tr,zeros(length(tr.output)), eta)
#     end
    

    
    
#     for _ in 1:step_md
#         run_md!(tr.net_f,T)
#         run_md!(tr.net_c,T)
#         # update gradient
#         update_Gradient!(tr, Gradient)
#         #update current free strains
#         for (ido,(edge,_,_,l0)) in enumerate(tr.output)
#             nodei, nodej = tr.net_f.edges[edge]
#             dist=norm(tr.net_f.pts[nodej, :] .- tr.net_f.pts[nodei, :])
#             sf_new[ido] += (dist - l0)/l0
#         end
#     end

#     Gradient=Gradient ./ step_md
#     sf_new .= sf_new ./ step_md
    
#     # stiffness update
#     learn_k!(tr, Gradient, alpha)
   
#     return sf_new # return current free strains on output edges 
# end

