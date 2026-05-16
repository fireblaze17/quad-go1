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
Stage 1 - standing policy, flat terrain, fixed friction=0.8

Status:
  accepted flat-ground standing baseline

Current baseline:
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

Next:
  - checkpoint this standing baseline
  - move to flat-ground friction randomization
  - keep contact diagnostics available while robustness is added
```

## Latest Decision: Flat-Ground Standing v2

**Status:** Accepted.

The policy now stands upright on flat Chrono terrain at fixed friction 0.8
without the earlier one-leg-up stance or visible vibration. The key fix was not
another pure smoothness reward. The policy and contact diagnostics showed that
leg-link collisions and weak foot-contact incentives were letting the robot use
bad support modes.

The accepted standing reward is:

```python
reward = (
    alive_bonus
    + upright_reward
    - pose_penalty
    - control_penalty
    - joint_vel_penalty
    - action_rate_penalty
    - tilt_penalty
    - ang_vel_penalty
    - xz_vel_penalty
    - foot_contact_penalty
)
```

Current weights:

```text
alive_bonus             1.00
upright_reward          0.15 * upright_score
pose_penalty            0.30 * mean(joint_error^2)
control_penalty         0.03 * mean(action^2)
joint_vel_penalty       0.01 * mean(joint_velocity^2)
action_rate_penalty     0.03 * mean(action_delta^2)
tilt_penalty            0.25 * (trunk_x_up^2 + trunk_y_up^2)
angular_vel_penalty     0.01 * mean(trunk_angular_velocity^2)
xz_vel_penalty          0.20 * mean([vx, vz]^2)
foot_contact_penalty    0.10 * mean(missing_foot_load^2)
minimum foot load       20 N per foot before penalty is zero
```

`leg_symmetry_error` is still logged as a diagnostic, but it is not an active
reward penalty in the accepted baseline.

Latest accepted evaluation:

```text
survival_rate:       1.000
mean_length:         1000.0
min_trunk_y:         0.337
min_upright_score:   1.000
mean_abs_xz_vel:     0.007
mean_abs_joint_vel:  0.393
mean_abs_action:     0.304
mean_foot_load:      32.09 N
min_foot_load:       17.58 N
termination:         truncated only
```

The viewer showed all four feet close to the same height, no calf/thigh/hip
contact load, low X/Z drift, and no obvious vibration.

## Why The Final Fix Worked

The important chain was:

1. Zero action was already stable, so the basic home pose and spawn height were
   not the current failure.
2. Policy viewer foot diagnostics showed foot displacement and foot velocity,
   meaning the shuffling was physically real, not only visual.
3. Link-contact diagnostics showed huge calf/thigh contact loads when those
   links were collidable. The policy could lean on non-foot collision geometry.
4. Matching the MaGIC-style contact setup, we disabled hip/thigh/calf terrain
   collisions and kept trunk + feet collidable.
5. After that, the old policy revealed the real issue: it could stand while
   leaving one foot unloaded.
6. A weak four-foot support penalty fixed the stance because it asks for contact
   load, not equal world foot height. That is better for future SCM terrain.

Tradeoff: disabling leg-link terrain collision is less literal for falls and
scrapes, but it is the better training model for standing and walking because
the learned policy should support itself through the feet. Trunk collision stays
enabled so falls still contact the terrain.

## Project Shape

```text
go1_env.py                 Chrono Gymnasium environment
view_env.py                zero-action/live test harness
train_stand.py             PPO standing-policy training
evaluate_stand.py          headless policy evaluation
view_stand_policy.py       trained-policy viewer with contact diagnostics
models/go1/go1_chrono.urdf Chrono-specific Go1 URDF
chrono_go1_soil.py         SCM deformable terrain milestone
mujoco/                    MuJoCo baseline (reference only)
docs/                      decision logs and roadmap
```

## Quick Start

Create the documented environment:

```powershell
C:\Users\ankus\anaconda3\Scripts\conda.exe env create -f environment.yml
```

```powershell
# View environment with zero policy action
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe view_env.py

# Train standing policy
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000

# Evaluate
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe evaluate_stand.py runs/stand/final_model.zip

# View trained policy
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe view_stand_policy.py runs/stand/final_model.zip
```

## Document Map

Read in this order if you are new to the project:

- [docs/reproducibility.md](docs/reproducibility.md) - install steps,
  commands, accepted metrics, and non-determinism notes
- [docs/experiments/standing_v2.md](docs/experiments/standing_v2.md) - model
  card for the accepted flat-ground standing baseline
- [docs/training_roadmap.md](docs/training_roadmap.md) - reward history,
  diagnostics, what worked, and what did not
- [docs/chrono_port_notes.md](docs/chrono_port_notes.md) - Chrono import,
  solver, contact, home pose, and reset decisions
- [docs/collision_debug_log.md](docs/collision_debug_log.md) - collision and
  contact debugging trail

## Technical Decisions

### Y-Up World + Position Control

Go1 source assets are ROS/Z-up, but this project uses Chrono with Y as the world
up direction. The imported robot root is rotated -90 degrees about X. All joints
use position control. Zero action means "hold the home pose," not "motors off."

Actions are normalized offsets:

```python
target = home + 0.20 * action
```

### Stable Chrono Home Pose

The Menagerie crouch (`hip=0.0`, `thigh=0.9`, `calf=-1.8`) slowly sank in this
Chrono import. The accepted Chrono baseline is:

```text
home pose:    hip=0.0, thigh=0.7, calf=-1.4
spawn height: 0.34 m
```

This pose starts at its natural support height and holds with zero action.

### MaGIC-Style Contact Setup

The current rigid-contact setup follows the relevant MaGIC 2025 Chrono lessons:

```text
solver:             BARZILAIBORWEIN
solver iterations:  60
ground friction:    per-episode friction range, currently fixed at 0.8
ground restitution: 0.1
ground Kn/Gn:       2e5 / 60
foot friction:      0.9
foot restitution:   0.01
foot Gn:            60
```

This is not a full clone of the MaGIC Go2 tutorial. This project uses Go1, a
different reward stack, and a standing-first training path.

### Collision Whitelist

Current training collision bodies:

```python
_ROBOT_COLLISION_BODIES = (
    "trunk",
    "FR_foot", "FL_foot", "RR_foot", "RL_foot",
)
```

Hips, thighs, calves, rotors, camera bodies, and sensor marker bodies do not
collide with the terrain. This prevents the policy from using the side of a leg
as a hidden support. The viewer still reports calf/thigh/hip contact loads so we
can catch accidental regressions.

## Roadmap

```text
Stage 1  train_stand.py       flat terrain, fixed friction=0.8       <- accepted
Stage 2  train_stand.py       flat terrain, randomized friction      <- next
Stage 3  train_walk.py        flat terrain walking
Stage 4  train_walk_scm.py    SCM deformable terrain fine-tuning
Stage 5  rollout collection   learned standing/walking skills
Stage 6  world model          obs/action/next_obs prediction
Stage 7  hierarchy            skill selection and planning
```

Immediate next steps:

```text
1. keep the current flat-ground standing model as the v2 baseline
2. run friction-randomized standing on flat rigid terrain
3. keep checking foot load, foot slip, and non-foot contact diagnostics
4. move to walking only after randomized standing survives cleanly
5. move to SCM after rigid terrain robustness is believable
```

## Detailed Notes

Future training runs write two metadata files next to the saved policy:

```text
args.json
env_constants.json
```

Keep those with any checkpoint that gets reported or shared.
