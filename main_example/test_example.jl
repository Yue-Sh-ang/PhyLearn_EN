using PhyLearn_EN
using StableRNGs
#train info
dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])

traintime=parse(Int, ARGS[6])
alpha=parse(Float64, ARGS[7])
timewindow=parse(Int, ARGS[8])



#test info
testT=parse(Float64, ARGS[9])
strain_source=parse(Float64, ARGS[10])
strain_target=parse(Float64, ARGS[11])#new input
seed2=parse(Int, ARGS[12])



record_per = 100000 #500tau
n_frames= 2000

root = "/path/to/your/project" 
net_file =  joinpath(root,"dim$(dim)","network$(network_id)","network.txt")
task_path = joinpath(root,"dim$(dim)","network$(network_id)","task$(taskid)")
if !isfile(netfile)
    error("$netfile not found: you should generate net first")
end

enm=ENM(net_file)
input,output=load_task(task_path)
trainpath=joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_tw$(timewindow)","seed$(seed)")
load_k(enm, joinpath(trainpath, "k$(traintime).f64"))

#put strain
if strain_source != 0.0
     put_strain!(enm, input[1][1], strain_source)
else
    enm.k[input[1][1]]=0.0 #if there is no source strain, then put the stiffness to be 0
end

if strain_target != 0.0
     put_strain!(enm, output[1][1], strain_target)
else
    enm.k[output[1][1]]=0.0 #if there is no source strain, then put the stiffness to be 0
end

test_path=joinpath(trainpath, "MSM_testT$(testT)_time$(traintime)_strain$(strain_source)_$(strain_target)_seed$(seed2)/")
mkpath(test_path)
sout= Vector{Float32}(undef, n_frames)

rng=StableRNG(seed2)
for stepid in 1:n_frames
    run_md!(enm,testT,steps=record_per, rng=rng)
    sout[stepid] = PhyLearn_EN.cal_strain(enm, output[1][1])
end

#In case you also want to see the (queched) config

#enm_copy=deepcopy(enm)
#quench_fire!(enm_copy)
#save_pts(enm_copy,joinpath(test_path, "IS_step$(n_frames).f32"))
#       println("save IS!stepid=$(stepid),docuid=$(docuid)")
#save_pts(enm,joinpath(test_path, "CS_step$(n_frames).f32"))
#       println("save CS!")



if strain_target == 0.0
 open(joinpath(test_path, "strain.f32"), "w") do file
    write(file, Float32.(sout))
 end
end


