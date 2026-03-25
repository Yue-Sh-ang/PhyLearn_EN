using PhyLearn_EN
using StableRNGs
using Base.Threads
dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])

alpha=parse(Float64, ARGS[6])
trainsteps=parse(Int, ARGS[7])+1


timewindow=200


save_per=2_000_000
print_per=1000_000


println("Dimension: $(dim), Network ID: $(network_id), taskid: $(taskid), Training Temperature: $(trainT)")

root="/data2/shared/yueshang/julia/"
net_file =  "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/network.txt"
task_path = "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/task$(taskid)/"
net=ENM(net_file)
if !isdir(task_path)
    mkdir(task_path)
    generate_task(net,task_path; s_in=[0.2],s_out=[0.2],Distant=true,seed=taskid)
end
input,output=load_task(task_path)
train0=Trainer_CL(net,input,output)

#classical eta=1 training boundary condition
for op in train0.input
        set_edge_k!(train0.net_f,op[1],100.0)
        set_edge_l0!(train0.net_f,op[1],(op[2]+1)*op[3])
        set_edge_k!(train0.net_c,op[1],100.0)
        set_edge_l0!(train0.net_c,op[1],(op[2]+1)*op[3])'
end
for op in train0.output
        set_edge_k!(train0.net_c,op[1],100.0)
        set_edge_l0!(train0.net_c,op[1],(op[2]+1)*op[3])
end



# training loop
N_ens = Threads.nthreads()
println("Using $(N_ens) ensemble replicas")
ensemble = [deepcopy(train0) for _ in 1:N_ens]
rngs = [StableRNG(seed + i) for i in 1:N_ens]

trainpath=joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_Ne$(N_ens)","seed$(seed)")
mkpath(trainpath)
t0=time()
E_f_list = Float64[]
E_c_list = Float64[]
strain_list = Float64[]
step_list = Int[]
for stepid in 1:trainsteps
    Gradient=zeros(length(train0.net_f.edges))
    Gradients_local = [zeros(length(Gradient)) for _ in 1:N_ens]
    
    Threads.@threads for i in 1:N_ens
        local_grad = Gradients_local[i]
        tr = ensemble[i]
        rng = rngs[i]
        run_md!(tr.net_f,steps=200, trainT, rng=rng)
        run_md!(tr.net_c,steps=200, trainT, rng=rng)
        update_grad!(tr, local_grad)
        
    end
    
    for i in 1:N_ens
        Gradient .+= Gradients_local[i]
    end
    Gradient ./= N_ens
    update_k!(train0, Gradient, alpha)

    #synchronize parameters back to ensemble
    for i in 1:N_ens
        ensemble[i].net_f.k .= train0.net_f.k
        ensemble[i].net_c.k .= train0.net_c.k
    end
    #document the ensemble average of energys and the output strain
    if stepid%print_per==0
        E_f_all=0.0
        E_c_all=0.0
        strain_fo_all=0.0
        
        for i in 1:N_ens
            tr = ensemble[i]
            E_f_all+=cal_elastic_energy(tr.net_f)
            E_c_all+=cal_elastic_energy(tr.net_c)
            strain_fo_all+=cal_strain(tr.net_f,tr,output[1][1])# this is only for one output edge
        end
        push!(E_f_list, E_f_all/N_ens)
        push!(E_c_list, E_c_all/N_ens)
        push!(strain_list, strain_fo_all/N_ens)
        push!(step_list, stepid)
    end

     if stepid%save_per==0 || stepid==1
        save_k(train0.net_f, joinpath(trainpath, "k$(stepid).f64"))
        open(joinpath(trainpath, "Grad$(stepid).f64"),"w") do io
                write(io,Gradient)
        end
    end
    
end

timereal=time() - t0

npzwrite(joinpath(trainpath, "training_log.npz"),
    Dict(
        "step" => step_list,
        "E_f" => E_f_list,
        "E_c" => E_c_list,
        "strain" => strain_list,
        "train_time" =>timereal
       )
)

      

