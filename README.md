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
  accepted B-capable checkpoint, validated on randomized friction=0.6-1.0

Current/default baseline:
  checkpoint:    runs/stand_friction_ab_065_095/final_model.zip
  validated on:  randomized friction=0.6-1.0
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

Next:
  - train/evaluate friction C from the AB B-capable checkpoint
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

Current B-capable standing checkpoint is preserved here:

```text
runs/stand_friction_ab_065_095/final_model.zip
```

All accepted standing checkpoints run full 1000-step flat-ground episodes with
trunk + foot collision only. The B-capable checkpoint was trained on
`0.65-0.95` and accepted on `0.6-1.0` after eval, viewer, and headless
diagnostics showed clean generalization. Detailed reward, contact, and
experiment rationale live in `docs/`.

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
