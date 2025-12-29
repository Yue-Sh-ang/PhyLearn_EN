# to find allosteric sites in ENM
using Graphs
using SimpleWeightedGraphs
using Random
using LinearAlgebra


function build_graph(enm::ENM)
    g=SimpleWeightedGraph(enm.n)
    for (i,(u,v)) in enumerate(enm.edges)
        add_edge!(g,u,v,enm.l0[i])
    end
    return g
end

function spdist(g::SimpleWeightedGraph, u::Int, v::Int)
    ds = dijkstra_shortest_paths(g, u).dists
    return ds[v]  
end

# Average endpoint-to-endpoint distance between two edges e0=(i,j) and e1=(k,l)
function cal_edge_distance(g::SimpleWeightedGraph, edge1::Int, edge2::Int)
    u,v = edges(g)[edge1]
    k,l = edges(g)[edge2]
    dists = [
        spdist(g, u, k),
        spdist(g, u, l),
        spdist(g, v, k),
        spdist(g, v, l)
    ]
    return mean(dists)
end

function choose_new_edge(enm::ENM,strain::Float64; inout::Union{Vector{Tuple{Int,Float64,Float64}},nothing}=nothing,Distant=true)
    excluded_nodes=Set{Int}()
    #exclude nodes involved in input/output edges
    if inout!==nothing
        for ip in inout
            push!(excluded_nodes, enm.edges[ip[1]][1])
            push!(excluded_nodes, enm.edges[ip[1]][2])
        end
    end
    #exclude nodes with degree less than d_min
    degree=cal_degree(enm);d_min=enm.dim+2
    for node in 1:enm.n
        if degree[node]<d_min
            push!(excluded_nodes, node)
        end
    end

    edge_candidates=Vector{Int}()
    for (i, (u,v)) in enumerate(enm.edges)
        if !(u in excluded_nodes) && !(v in excluded_nodes)
            push!(edge_candidates, i)
        end
    end

    if length(edge_candidates)==0
        error("No available edges to choose from!")
    end

    if inout===nothing || !Distant
        selected_edge = edge_candidates[rand(1:length(edge_candidates))]
    else
        g=build_graph(enm)
        max_dist=-1.0
        selected_edge=-1
        for edge in edge_candidates
            min_dist=typemax(Float64)
            for ip in inout
                dist=cal_edge_distance(g, edge, ip[1])
                if dist<min_dist
                    min_dist=dist
                end
            end
            if min_dist>max_dist
                max_dist=min_dist
                selected_edge=edge
            end
        end
    end

    return (selected_edge,strain, enm.l0[selected_edge])
end

function generate_task(enm::ENM,dir::String;s_in::Vector{Float64}=[0.2],s_out::Vector{Float64}=[0.2],Distant=true)
    input=Vector{Tuple{Int,Float64,Float64}}()
    output=Vector{Tuple{Int,Float64,Float64}}()
    exist=nothing
    for s in s_in
        push!(input, choose_new_edge(enm,s; inout=exist,Distant=Distant))
        if exist===nothing
            exist=[input[end]]
        else
            push!(exist, input[end])
        end
    end
    for s in s_out
        push!(output, choose_new_edge(enm,s; inout=exist,Distant=Distant))
        push!(exist, output[end])
    end
    open(joinpath(dir, "input.txt"), "w") do io
        println(io, length(input))
        for ip in input
            println(io, "$(ip[1]) $(ip[2]) $(ip[3])")
        end
    end
    open(joinpath(dir, "output.txt"), "w") do io
        println(io, length(output))
        for op in output
            println(io, "$(op[1]) $(op[2]) $(op[3])")
        end
    end
    return input,output
end

function load_task(dir::String)
    input = Vector{Tuple{Int,Float64,Float64}}()
    output= Vector{Tuple{Int,Float64,Float64}}()

    open(joinpath(dir, "input.txt"), "r") do io
        n = parse(Int, readline(io))
        for _ in 1:n
            line = readline(io)
            parts = split(line)
            edge = parse(Int, parts[1])
            st   = parse(Float64, parts[2])
            l0   = parse(Float64, parts[3])
            push!(input, (edge,st,l0))
        end
    end

    open(joinpath(dir, "output.txt"), "r") do io
        n = parse(Int, readline(io))
        for _ in 1:n
            line = readline(io)
            parts = split(line)
            edge = parse(Int, parts[1])
            st   = parse(Float64, parts[2])
            l0   = parse(Float64, parts[3])
            push!(output, (edge,st,l0))
        end
    end

    return input,output
end