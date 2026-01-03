# To record locations of the markov state model
using PhyLearn_EN
using StableRNGs

dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])

testT=parse(Float64, ARGS[6])
strain_source=parse(Float64, ARGS[7])
seed2=parse(Int, ARGS[8])


traintime=10_000
alpha=50.0
timewindow=200

record_per = 5000 #50tau
n_frames= 2000


root="/data2/shared/yueshang/julia/"
net_file =  "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/network.txt"
task_path = "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/task$(taskid)/"

enm=ENM(net_file)
input,output=load_task(task_path)

load_k(enm, joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_tw$(timewindow)/seed$(seed)/k$(traintime).f64"))
n_soft=10
phi,lambda = mode_basis(enm, k=n_soft)

#put strain
put_strain!(enm, input[1][1], strain_source)

test_path=joinpath(task_path, "MSM_testT$(testT)_strain$(strain_source)_seed$(seed)/")
mkpath(test_path)
data    = Matrix{Float32}(undef, n_frames, n_soft)
sout= Vector{Float32}(undef, n_frames)
for stepid in 1:n_frames
    
    rng=StableRNG(seed2+stepid)
    run_md!(enm,testT,steps=record_per, rng=rng)
    data[stepid, :] .= project_modes_rigid(enm, phi)
    sout[stepid] = cal_strain(enm, output[1][1])

end

open(joinpath(test_path, "project10modes.f32"), "w") do file
    write(file, Float32.(data))
end

open(joinpath(test_path, "strain.f32"), "w") do file
    write(file, Float32.(sout))
end

TICA_=tica(data;lag=10,k=3)

open(join)