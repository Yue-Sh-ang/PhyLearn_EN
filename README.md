# PhyLearn_EN

Physical learning in elastic networks, written in Julia.

PhyLearn_EN trains elastic network models (ENMs) to perform **allosteric tasks**: straining a chosen *input* bond should produce a target strain on a distant *output* bond. Training uses **coupled learning** (CL) at finite temperature. Both the *free* and *clamped* states evolve under Langevin dynamics, and bond stiffnesses are updated by a local learning rule. The package also includes tools for testing trained networks and for analyzing their thermal fluctuations with a Markov state model (soft-mode projection → TICA → clustering).

## Features

- **Elastic network model** in 2D and 3D: harmonic springs, Langevin MD, FIRE energy minimization, Hessian and normal modes, and plotting.
- **Allosteric task generation**: picks input and output bonds that are far apart on the network graph (shortest-path distance weighted by rest length), and excludes poorly connected nodes.
- **Coupled learning trainer**: a stiffness-only learning rule, averaged over a time window of thermal MD.
- **MSM utilities**: rigid alignment (Kabsch), projection onto the softest non-trivial modes, and time-lagged ICA (TICA).
- **Python analysis helpers** for reading and visualizing outputs.

## Repository layout

```
PhyLearn_EN/
├── Project.toml / Manifest.toml   # Julia environment
├── src/
│   ├── PhyLearn_EN.jl   # module entry point and exports
│   ├── ENM.jl           # ENM struct, I/O, forces/energies, MD, FIRE, modes, plotting
│   ├── Allosteric.jl    # graph distances, input/output bond selection, task I/O
│   ├── CL.jl            # coupled-learning trainer and update rules
│   └── MSM.jl           # rigid alignment, soft-mode projection, TICA
├── main_example/
│   ├── train_example.jl # train a network on an allosteric task
│   └── test_example.jl  # run MD on a trained network and record output strain
└── analysis/
    └── visualize_utils.py  # Python helpers for loading and plotting results
```

## Installation

Requires Julia (developed with 1.11/1.12).

```bash
git clone https://github.com/Yue-Sh-ang/PhyLearn_EN.git
cd PhyLearn_EN
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run scripts inside the project environment:

```bash
julia --project=. main_example/train_example.jl <args...>
```

The Python helpers need `numpy`, `scipy` and `matplotlib`.

## Network file format

Networks are read from a plain-text file with `ENM("network.txt")`:

```
dim
n
x1 y1 (z1)
...
xn yn (zn)
ne
i1 j1 k1 l0_1
...
i_ne j_ne k_ne l0_ne
```

`dim` is 2 or 3, `n` is the number of nodes, and `ne` is the number of edges. Each edge line gives its two node indices (1-based), the stiffness `k`, and the rest length `l0`.

## Quick start

```julia
using PhyLearn_EN, StableRNGs

net = ENM("network.txt")

# choose one input and one output bond, each with target strain 0.2
input, output = generate_task(net, "task1"; s_in=[0.2], s_out=[0.2], Distant=true, seed=1)

# coupled-learning trainer
tr = Trainer_CL(net, input, output)

# boundary conditions: input clamped in both states, output clamped only in the clamped state
for (e, s, l0) in tr.input
    for nw in (tr.net_f, tr.net_c)
        set_edge_k!(nw, e, 100.0); set_edge_l0!(nw, e, (1 + s) * l0)
    end
end
for (e, s, l0) in tr.output
    set_edge_k!(tr.net_c, e, 100.0); set_edge_l0!(tr.net_c, e, (1 + s) * l0)
end

rng = StableRNG(1)
T, alpha, timewindow = 1e-4, 0.0625, 1200
for step in 1:10_000
    grad = zeros(length(tr.net_f.edges))
    for _ in 1:timewindow
        run_md!(tr.net_f, T; rng=rng)
        run_md!(tr.net_c, T; rng=rng)
        update_grad!(tr, grad)
    end
    update_k!(tr, grad, alpha)
end

save_k(tr.net_f, "k_trained.f64")
```

## Example scripts

Both scripts expect networks under a project root with this structure:

```
<root>/dim<dim>/network<network_id>/network.txt
<root>/dim<dim>/network<network_id>/task<taskid>/{input.txt, output.txt}
```

Set `root` at the top of each script before running.

### Training

```bash
julia --project=. main_example/train_example.jl dim network_id taskid trainT seed [alpha] [timewindow] [trainsteps]
```

| Argument | Meaning | Default |
|---|---|---|
| `dim` | spatial dimension (2 or 3) | required |
| `network_id` | which network to load | required |
| `taskid` | task index; also used as the seed when generating the task | required |
| `trainT` | training temperature | required |
| `seed` | RNG seed for MD | required |
| `alpha` | learning rate | `0.0625` |
| `timewindow` | MD steps per gradient update | `1200` |
| `trainsteps` | number of learning updates | `16_000_000` |

If the task folder doesn't exist yet, the script generates it. Stiffnesses are saved every `save_per` steps to:

```
task<taskid>/trainT<T>_alpha<alpha>_tw<timewindow>/seed<seed>/k<step>.f64
```

### Testing

```bash
julia --project=. main_example/test_example.jl dim network_id taskid trainT seed traintime alpha timewindow testT strain_source strain_target seed2
```

This script loads the stiffnesses saved at step `traintime`, applies the source strain (and optionally a target strain), runs MD at `testT`, and records the output-bond strain to `strain.f32` inside an `MSM_testT…` folder.

## Main API

| Area | Functions |
|---|---|
| Network | `ENM`, `save_enm`, `add_edge!`, `cal_degree`, `reset_config!` |
| Dynamics | `run_md!` (Langevin), `quench_fire!` (FIRE minimization) |
| Observables | `cal_elastic_energy`, `cal_kinetic_energy`, `cal_strain`, `put_strain!` |
| Modes | `cal_elastic_jacobian`, `cal_modes` |
| I/O | `save_k`, `load_k`, `save_pts`, `load_pts` |
| Plotting | `plot_net`, `plot_displacement` |
| Tasks | `generate_task`, `load_task`, `choose_new_edge`, `build_graph`, `cal_edge_distance` |
| Training | `Trainer_CL`, `set_edge_k!`, `set_edge_l0!`, `update_grad!`, `update_k!`, `load_trainer_CL` |
| MSM | `rigid_align!`, `mode_basis`, `project_modes_rigid`, `tica` |

## License

MIT. See [LICENSE](LICENSE).
