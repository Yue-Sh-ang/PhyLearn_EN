# To record locations of the markov state model
using PhyLearn_EN
using StableRNGs
#train info
dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])
#test info
testT=parse(Float64, ARGS[6])
strain_source=parse(Float64, ARGS[7])
seed2=parse(Int, ARGS[8])


traintime=10_000
alpha=50.0
timewindow=200

record_per = 20000 #100tau
n_frames= 2000


root="/data2/shared/yueshang/julia/"
net_file =  "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/network.txt"
task_path = "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/task$(taskid)/"

enm=ENM(net_file)
input,output=load_task(task_path)
trainpath=joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_tw$(timewindow)","seed$(seed)")
load_k(enm, joinpath(trainpath, "k$(traintime).f64"))
n_soft=10
phi,lambda = PhyLearn_EN.mode_basis(enm, k=n_soft)

#put strain
if strain_source != 0.0
     put_strain!(enm, input[1][1], strain_source)
end


test_path=joinpath(trainpath, "MSM_testT$(testT)_strain$(strain_source)_seed$(seed)/")
mkpath(test_path)
data    = Matrix{Float32}(undef, n_frames, n_soft)
sout= Vector{Float32}(undef, n_frames)
for stepid in 1:n_frames
    
    rng=StableRNG(seed2+stepid)
    run_md!(enm,testT,steps=record_per, rng=rng)
    data[stepid, :] .= PhyLearn_EN.project_modes_rigid(enm, phi)
    sout[stepid] = PhyLearn_EN.cal_strain(enm, output[1][1])

end

open(joinpath(test_path, "data10modes.f32"), "w") do file
    write(file, Float32.(data))
end

open(joinpath(test_path, "strain.f32"), "w") do file
    write(file, Float32.(sout))
end

