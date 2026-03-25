using PhyLearn_EN
using StableRNGs
using NPZ
dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])

alpha=parse(Float64, ARGS[6])
trainsteps=parse(Int, ARGS[7])+1


timewindow=200


save_per=100_000
print_per=20_000



root="/vast/projects/ajliu/liu/yueshang/data/PhyLearn_EN"
net_file =  joinpath(root,"dim$(dim)","network$(network_id)","network.txt")
task_path = joinpath(root,"dim$(dim)","network$(network_id)","task$(taskid)")

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
trainpath=joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_tw$(timewindow)","seed$(seed)")
rng=StableRNG(seed)
mkpath(trainpath)
E_f_list = Float64[]
E_c_list = Float64[]
strain_list = Float64[]
step_list = Int[]

t0=time()
for stepid in 1:trainsteps
    Gradient=zeros(length(train0.net_f.edges))
    for sid in 1:timewindow 
        run_md!(train0.net_f,trainT,rng=rng)
        run_md!(train0.net_c,trainT,rng=rng)
        update_grad!(train0, Gradient)
    end
    update_k!(train0, Gradient, alpha)
    if stepid%print_per==0 || stepid==1
        E_f=cal_elastic_energy(train0.net_f)
        E_c=cal_elastic_energy(train0.net_c)
        strain_f_out=cal_strain(train0.net_f,train0.output[1][1])# this is only for one output edge
        push!(E_f_list, E_f)
        push!(E_c_list, E_c)
        push!(strain_list, strain_f_out)
        push!(step_list, stepid)
    end

    if stepid%save_per==0 || stepid==1
        save_k(train0.net_f, joinpath(trainpath, "k$(stepid).f64"))
        save_pts(train0.net_f, joinpath(trainpath, "pts$(stepid).f64"))
        #Also document the gradient
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

