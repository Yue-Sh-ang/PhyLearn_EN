using PhyLearn_EN
using StableRNGs
using NPZ

dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])

#training hyper-parameters (optional; defaults used if not given)
alpha      = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.0625
timewindow = length(ARGS) >= 7 ? parse(Int,     ARGS[7]) : 1200
trainsteps = (length(ARGS) >= 8 ? parse(Int,    ARGS[8]) : 16_000_000) + 1  #CPU hour= 11h


save_per=1_000_000
print_per=200_000


#CHANGE BEFORE USE
root = "/path/to/your/project"
net_file =  joinpath(root,"dim$(dim)","network$(network_id)","network.txt")
if !isfile(netfile)
    error("$netfile not found: you should generate net first")
end #I generate nets before tune them

task_path = joinpath(root,"dim$(dim)","network$(network_id)","task$(taskid)")


net=ENM(net_file)
if !isdir(task_path)
    mkdir(task_path)
    generate_task(net,task_path; s_in=[din],s_out=[dout],Distant=true,seed=taskid)
end

input,output=load_task(task_path)
#coupled learning trainer
train0=Trainer_CL(net,input,output)

#set boundary conditions for net_f and net_c
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

if !isdir(trainpath)
    mkpath(trainpath)
end

rng=StableRNG(seed)

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
        strain_f_out=cal_strain(train0.net_f,train0.output[1][1])#output strain of free states
        push!(E_f_list, E_f)
        push!(E_c_list, E_c)
        push!(strain_list, strain_f_out)
        push!(step_list, stepid)
    end

   if stepid%save_per==0:
    
        save_k(train0.net_f, joinpath(trainpath, "k$(stepid).f64"))
        #save_pts(train0.net_f, joinpath(trainpath, "pts$(stepid).f64"))
	#save_pts(train0.net_c, joinpath(trainpath, "pts_c$(stepid).f64"))
	
	#open(joinpath(trainpath, "Grad$(stepid).f64"),"w") do io
	#	write(io,Gradient)
	#end
    end
    
end

timereal=time() - t0

#npzwrite(joinpath(trainpath, "train_log.npz"),
#    Dict(
#        "step" => step_list,
#        "E_f" => E_f_list,
#        "E_c" => E_c_list,
#        "strain" => strain_list,
#        "CPU_time" =>timereal
#       )
#)

