using PhyLearn_EN
using StableRNGs
dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])

alpha=50.0
timewindow=200
trainsteps=10_001

save_per=1_000
print_per=200


println("Dimension: $(dim), Network ID: $(network_id), taskid: $(taskid), Training Temperature: $(trainT)")

root="/data2/shared/yueshang/julia/2d/"
net_file = joinpath(root, "/dim$(dim)/network$(network_id)/network.txt")
task_path = joinpath(root, "/dim$(dim)/network$(network_id)/task$(taskid)/")
net=ENM(net_file)
if !isdirectory(task_path)
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
trainpath=joinpath(task_path, "/trainT$(trainT)_alpha$(alpha)_tw$(timewindow)/seed$(seed)/")
rng=StableRNG(seed)
mkpath(trainpath)
t0=time()
for stepid in 1:trainsteps
    Gradient=zeros(length(train0.net_f.edges))
    for sid in 1:timewindow 
        run_md!(train0.net_f,trainT,rng=rng)
        run_md!(train0.net_c,trainT,rng=rng)
        update_grad!(train0, Gradient)
    end
    update_k!(train0, Gradient, alpha)
    if stepid%print_per==0
        E_f=cal_elastic_energy(train0.net_f)
        E_c=cal_elastic_energy(train0.net_c)
        strain_f_out=cal_strain(train0.net_f,train0.output[1][1])# this is only for one output edge
        println("Step: $(stepid), E_f: $(E_f), E_c: $(E_c), strain: $(strain_f_out)")
    end

    if stepid%save_per==0
        save_k(train0.net_f, joinpath(trainpath, "k$(stepid).f64"))
        save_pts(train0.net_f, joinpath(trainpath, "pts$(stepid).f64"))
        
    end
    
end
println("trainingtime = ", time() - t0, " s")

