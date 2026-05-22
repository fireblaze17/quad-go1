# Quad Go1

Project Chrono robotics simulation and ML project for a Unitree Go1-style
quadruped.

## Goal

1. Build a stable Go1 simulation in Chrono.
2. Wrap it as a Gymnasium environment.
3. Train standing, then locomotion policies.
4. Transfer to Chrono SCM deformable terrain.
5. Collect rollouts, train a world model, and add hierarchical skill selection.

## Current Status

```text
Stage 2 - standing policy, flat terrain, randomized friction

Status:
  accepted fixed-friction baseline
  accepted friction A baseline, randomized friction=0.7-0.9
  accepted AB checkpoint, validated on randomized friction=0.6-1.0
  AB checkpoint also passed friction C=0.5-1.1 eval/view/diagnosis
  trained C-from-AB challenger rejected for drift and asymmetric support
  equal-budget scratch C seeds 1/2/3 rejected for foot-unload/load-bias failures

Current/default baseline:
  checkpoint:    runs/stand_friction_ab_065_095/final_model.zip
  validated on:  randomized friction=0.6-1.0 and stress-tested on 0.5-1.1
  decision:      official current standing baseline after C-range comparison
  home pose:     [hip=0.0, thigh=0.7, calf=-1.4] per leg
  spawn height:  0.34 m
  action scale:  0.20 rad normalized offset
  collision:     trunk + feet only
  contact:       MaGIC-style Chrono rigid contact settings
  reward:        upright survival + smoothness + four-foot support

Solved:
  - Chrono simulation + Y-up world
  - Chrono-specific Go1 URDF with trunk as the free root
  - Joint angle observation sign/axis bug
  - Zero-overhead home-pose spawn with DoAssembly
  - Stable Chrono home pose and spawn height
  - Non-foot leg collision support exploit
  - Raised/unloaded front-left foot stance
  - Visible vibration after the contact/support fix
  - WSL Ubuntu runtime after native Windows Smart App Control blocked PyChrono DLLs
  - Friction A randomized-friction continuation
  - B-range generalization from the AB checkpoint without extra B fine-tuning
  - C-range generalization from the same AB checkpoint without accepting C fine-tuning

Next:
  - keep AB as the official standing baseline before reset-noise work
  - use run_regression.py before accepting any future standing challenger
  - use headless diagnostics before any further standing fine-tuning
  - keep fixed-friction 0.8 regression checks
  - keep contact diagnostics available while robustness widens
```

## Accepted Baselines

Fixed-friction standing v2 is preserved here:

```text
runs/stand_base_v2/final_model.zip
```

Current friction A standing baseline is preserved here:

```text
runs/stand_friction_a_07_09/final_model.zip
```

Current official standing baseline is preserved here:

```text
runs/stand_friction_ab_065_095/final_model.zip
```

All accepted standing checkpoints run full 1000-step flat-ground episodes with
trunk + foot collision only. The official current baseline was trained on
`0.65-0.95`, accepted on `0.6-1.0`, and then stress-tested successfully on
the wider C range `0.5-1.1`. The explicitly trained C-from-AB continuation is
not accepted because it survived by drifting/sliding more and developing a
left/right load-bias pattern. Detailed reward, contact, and experiment
rationale live in `docs/`.

Official decision: `runs/stand_friction_ab_065_095/final_model.zip` remains the
current/default baseline. It is better than the trained C-from-AB checkpoint and
all three equal-budget scratch-C seeds on clean standing quality, not just
survival.

`runs/` is intentionally gitignored, so accepted checkpoints are not stored in
GitHub. A fresh clone can run the code and train from scratch, but reproducing
the accepted numbers above requires copying these checkpoint folders into
`runs/` out-of-band.

Required accepted checkpoints:

```text
runs/stand_base_v2/final_model.zip
runs/stand_friction_a_07_09/final_model.zip
runs/stand_friction_ab_065_095/final_model.zip
```

## Project Shape

```text
go1_env.py                 Chrono Gymnasium environment
view_env.py                zero-action/live test harness
train_stand.py             PPO standing-policy training
evaluate_stand.py          headless policy evaluation
view_stand_policy.py       trained-policy viewer with contact diagnostics
friction_curriculum.py     flat randomized-friction curriculum helper
project_config.py          shared paths and runtime defaults
diagnose_policy.py         headless tilt/contact diagnosis
run_regression.py          eval + diagnosis regression runner
models/go1/go1_chrono.urdf Chrono-specific Go1 URDF
chrono_go1_soil.py         SCM deformable terrain milestone
mujoco/                    MuJoCo baseline (reference only)
docs/                      decision logs and roadmap
```

## Quick Start

The active development environment is WSL Ubuntu, not native Windows. Create
and activate the conda environment inside WSL:

```bash
conda env create -f environment.yml
conda activate chrono-go1
```

After activation, use `python` for project commands. On Ankus's WSL machine,
the equivalent explicit interpreter is:

```bash
/home/ankus/miniforge3/envs/chrono-go1/bin/python
```

```bash
# View environment with zero policy action
python view_env.py

# Train standing policy
python train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000

# Evaluate
python evaluate_stand.py runs/stand/final_model.zip

# Run eval + diagnosis and save a compact summary
python run_regression.py runs/stand_friction_ab_065_095/final_model.zip --name ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100

# View trained policy
python view_stand_policy.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8

# Prepare and inspect the flat friction curriculum commands
python friction_curriculum.py prepare-base
python friction_curriculum.py all-commands
```

## Current Friction A Commands

Friction A is the accepted first randomized-friction stage. The accepted
checkpoint is:

```text
runs/stand_friction_a_07_09/final_model.zip
```

```bash
# Train friction A from the accepted base for 300k steps
python train_stand.py --terrain flat --friction-min 0.7 --friction-max 0.9 --load runs/stand_base_v2/final_model.zip --save-dir runs/stand_friction_a_07_09 --timesteps 300000

# Evaluate friction A across its randomized range
python evaluate_stand.py runs/stand_friction_a_07_09/final_model.zip --terrain flat --friction-min 0.7 --friction-max 0.9 --episodes 10

# View friction A across its randomized range
python view_stand_policy.py runs/stand_friction_a_07_09/final_model.zip --terrain flat --friction-min 0.7 --friction-max 0.9

# Sanity-check friction A at fixed 0.8
python evaluate_stand.py runs/stand_friction_a_07_09/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10
```

`view_stand_policy.py` defaults to the accepted B-capable AB checkpoint and
`0.6-1.0` friction range for VS Code Run-button use. Pass an explicit model path
and friction flags when viewing older checkpoints.

## Current B-Capable Checkpoint

The clean AB checkpoint generalizes to the full B friction range and is accepted
as the current B-capable standing policy:

```text
runs/stand_friction_ab_065_095/final_model.zip
```

It was trained on `0.65-0.95`, then accepted on `0.6-1.0` because it survived
30/30 episodes, stayed visually upright in the viewer, kept non-foot loads at
zero, and did not enter the learned FL-heavy lean seen after additional B
fine-tuning.

```bash
# Evaluate AB on full B randomized range
python evaluate_stand.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0 --episodes 30

# View AB on full B randomized range
python view_stand_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0

# Headless tilt/contact diagnosis
python diagnose_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0 --episodes 30 --out diagnostics/ab_on_b_range
```

Do not keep training on B just to match the folder name. `friction_curriculum.py`
now treats B as accepted via AB and uses AB as the load checkpoint for friction
C. The tested B continuations survived but learned a left/right load-bias
attractor: FL carried too much load, opposite/right-side feet unloaded first,
and visible lean followed. Use diagnostics before changing reward or physics.

## Current C-Range Finding

Official current baseline after C comparison:

```text
runs/stand_friction_ab_065_095/final_model.zip
```

Five C-range comparisons now exist. All were evaluated on flat terrain with
randomized friction `0.5-1.1` for 100 episodes. Survival alone is not enough:
the standing baseline must also stay upright, avoid steady sliding, keep clean
four-foot support, and avoid a repeatable load-bias attractor.

| Model family | Training path | Timesteps | Survival | Upright | Contact error | Min foot load | X/Z drift | Diagnosis pattern | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AB-on-C | fixed 0.8 -> A -> AB, tested on C | 700k staged | 1.000 | 0.999 | 0.009350 | 23.431 N | 0.011950 | no tilt crossing; mostly diagonal load | accepted/current baseline |
| C-from-AB | fixed 0.8 -> A -> AB -> C | 1.0M staged | 1.000 | 0.999 | 0.016239 | 20.134 N | 0.045557 | no tilt crossing, but FL-heavy left/right bias | rejected: drift/sliding and asymmetric support |
| Scratch C seed1 | random init -> C | 700k | 1.000 | 0.996 | 0.016500 | 19.784 N | 0.024861 | foot unload before tilt in 100/100; FL-heavy | rejected: worse contact/upright/load balance |
| Scratch C seed2 | random init -> C | 700k | 1.000 | 0.987 | 0.104047 | 15.162 N | 0.044395 | foot unload before tilt in 100/100; left/right bias | rejected: large contact/upright regression |
| Scratch C seed3 | random init -> C | 700k | 1.000 | 0.974 | 0.249241 | 5.966 N | 0.039411 | foot unload before tilt in 100/100; FL unloaded | rejected: severe contact/upright regression |

Run the C comparison battery with:

```bash
python run_regression.py runs/stand_friction_ab_065_095/final_model.zip --name ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_c_05_11/final_model.zip --name c_from_ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_c_scratch_seed1_700k/final_model.zip --name c_scratch_seed1_700k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_c_scratch_seed2_700k/final_model.zip --name c_scratch_seed2_700k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_c_scratch_seed3_700k/final_model.zip --name c_scratch_seed3_700k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100

# Viewer still matters before accepting a challenger.
python view_stand_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.5 --friction-max 1.1
```

AB-on-C reference signature:

```text
survival_rate:       1.000
mean_length:         1000.0
mean_reward:         1138.815
min_upright_score:   0.999
foot_contact_error:  0.009350
min_foot_load:       23.431 N
mean_abs_xz_vel:     0.011950
max_abs_xz_vel:      0.019967
diagnosis:           no_tilt_threshold_crossing in 100/100 episodes
max_nonfoot_load:    0.000000
```

The first trained C-from-AB challenger at
`runs/stand_friction_c_05_11/final_model.zip` survived 100/100 episodes, but
is rejected for now. It had worse drift/contact quality than AB:

```text
foot_contact_error:  0.016239
min_foot_load:       20.134 N
mean_abs_xz_vel:     0.045557
max_abs_xz_vel:      0.064314
leg_symmetry_error:  0.003477
diagnosis:           FL dominant-loaded in 85/100, RR least-loaded in 79/100
max_foot_dxz:        0.416218
```

The first equal-budget direct scratch C challenger used the same total budget as
the successful curriculum path (`700k` timesteps). It also survived 100/100
episodes, but it is rejected as a challenger to AB:

```text
model:               runs/stand_friction_c_scratch_seed1_700k/final_model.zip
mean_reward:         1136.552
min_upright_score:   0.996
foot_contact_error:  0.016500
min_foot_load:       19.784 N
mean_abs_xz_vel:     0.024861
max_abs_xz_vel:      0.032744
tilt_error:          0.002643
leg_symmetry_error:  0.002757
diagnosis:           foot_unload_before_tilt in 100/100 episodes
load bias:           FL dominant-loaded in 97/100, left_vs_right in 100/100
```

This is the same family of failure seen in rejected B continuations and the
rejected C-from-AB run: the policy survives but uses an asymmetric support mode.
Scratch-C seeds 2 and 3 made the point stronger, not weaker: both survived all
100 episodes, but both had much worse upright/contact scores and repeated
`foot_unload_before_tilt` in 100/100 episodes. Keep AB as the current/default
baseline unless a future experiment beats the AB reference on eval, diagnosis,
and viewer behavior.

Native Windows PyChrono was abandoned for this project because Windows Smart
App Control blocked unsigned Chrono extension DLLs such as `Chrono_vehicle.dll`
and `_parsers.pyd`. If you hit that import failure, use WSL Ubuntu and the WSL
conda environment instead. See [docs/reproducibility.md](docs/reproducibility.md)
for the detailed recovery notes.

## Document Map

Read in this order if you are new to the project:

- [docs/reproducibility.md](docs/reproducibility.md) - install steps,
  commands, accepted metrics, and non-determinism notes
- [docs/experiments/standing_v2.md](docs/experiments/standing_v2.md) - model
  card for the accepted flat-ground standing baseline
- [docs/experiments/friction_curriculum.md](docs/experiments/friction_curriculum.md) -
  randomized-friction curriculum stages and commands
- [docs/training_roadmap.md](docs/training_roadmap.md) - reward history,
  diagnostics, what worked, and what did not
- [docs/chrono_port_notes.md](docs/chrono_port_notes.md) - Chrono import,
  solver, contact, home pose, and reset decisions
- [docs/collision_debug_log.md](docs/collision_debug_log.md) - collision and
  contact debugging trail

## Roadmap

```text
Stage 1  train_stand.py       flat terrain, fixed friction=0.8       <- accepted
Stage 2  train_stand.py       flat terrain, friction A=0.7-0.9       <- accepted
Stage 2b train_stand.py       flat terrain, friction B=0.6-1.0       <- accepted via AB generalization
Stage 2c evaluation           flat terrain, friction C=0.5-1.1       <- AB accepted; C-from-AB and scratch C seeds rejected
Stage 3  train_walk.py        flat terrain walking
Stage 4  train_walk_scm.py    SCM deformable terrain fine-tuning
Stage 5  rollout collection   learned standing/walking skills
Stage 6  world model          obs/action/next_obs prediction
Stage 7  hierarchy            skill selection and planning
```

Training runs write two metadata files next to the saved policy:

```text
args.json
env_constants.json
```

Keep those with any checkpoint that gets reported or shared.

Shared paths and defaults, including the current baseline and SB3 CPU device,
live in `project_config.py`.
