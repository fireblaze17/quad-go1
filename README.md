# Quad Go1

Project Chrono robotics simulation and ML project for a Unitree Go1-style
quadruped.

## Goal

1. Build a stable Go1 simulation in Chrono.
2. Wrap it as a Gymnasium environment.
3. Train standing then locomotion policies.
4. Transfer to Chrono SCM deformable terrain.
5. Collect rollouts, train a world model, add hierarchical skill selection.

## Current Status

```text
Stage 1 - standing policy, flat terrain, fixed friction=0.8

Active reward:
  alive + upright
  - pose
  - control
  - joint_velocity
  - action_rate
  - tilt
  - angular_velocity
  - xz_velocity

Solved:
  - Chrono simulation + Y-up world
  - Chrono-specific Go1 URDF with trunk as the free root
  - Hip joint axis bug; hips are visible to pose penalty
  - Joint observations fixed; motor frames report true home pose
  - Motor ramp removed; DoAssembly places the robot at home pose with no warm-up
  - Stable Chrono home pose: hip=0.0, thigh=0.7, calf=-1.4
  - Spawn height 0.34 m matches the less-crouched support height
  - Flat-ground standing v1 completes 1000-step fixed-friction evaluations

Pending:
  - Patch remaining mild in-place shuffling
  - Save this as the flat-ground standing v1 checkpoint
  - Start Stage 2 with randomized flat-ground friction
```

## Latest Decision: Flat-Ground Standing v1

**Status:** Accepted as the first usable standing baseline.

The current policy stands on flat Chrono terrain at fixed friction 0.8. It no
longer sinks, tips, or develops the large one-sided lean that appeared during
earlier reward tests. The remaining defect is mild in-place leg chatter.

The accepted reward stack is:

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
)
```

Current weights:

```text
alive_bonus             1.00
upright_reward          0.15 * upright_score
pose_penalty            0.10 * mean(joint_error^2)
control_penalty         0.03 * mean(action^2)
joint_vel_penalty       0.01 * mean(joint_velocity^2)
action_rate_penalty     0.01 * mean(action_delta^2)
tilt_penalty            0.25 * (trunk_x_up^2 + trunk_y_up^2)
angular_vel_penalty     0.01 * mean(trunk_angular_velocity^2)
xz_vel_penalty          0.20 * mean([vx, vz]^2)
```

Latest accepted evaluation:

```text
survival_rate:       1.000
mean_length:         1000.0
min_trunk_y:         0.336
min_upright_score:   0.994
mean_abs_xz_vel:     0.040
mean_abs_joint_vel:  0.283
mean_abs_action:     0.380
termination:         truncated only
```

**Tradeoff:** this is stable enough to checkpoint, but not polished. The policy
still uses visible corrective leg motion. The next reward work should target
smoothness one term at a time so we do not break the recovered upright baseline.

## Project Shape

```text
go1_env.py                 Chrono Gymnasium environment
view_env.py                zero-action/live test harness
train_stand.py             PPO standing-policy training
evaluate_stand.py          headless policy evaluation
view_stand_policy.py       trained-policy viewer
models/go1/go1_chrono.urdf Chrono-specific Go1 URDF
chrono_go1_soil.py         SCM deformable terrain milestone
mujoco/                    MuJoCo baseline (reference only)
docs/                      decision logs and roadmap
```

## Quick Start

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

## Technical Decisions

Each section records what forced the decision, what was chosen, and what it
costs. Full ADRs:
[docs/chrono_port_notes.md](docs/chrono_port_notes.md) -
[docs/training_roadmap.md](docs/training_roadmap.md) -
[docs/collision_debug_log.md](docs/collision_debug_log.md)

### ADR-001: Y-Up World + Position Control

**Status:** Accepted

Go1 source assets are ROS/Z-up, but this project uses Chrono with Y as the world
up direction. The imported robot root is rotated -90 degrees about X. All joints
use position control. Zero action means "hold the home pose," not "motors off."

Actions are normalized offsets:

```python
target = home + 0.25 * action
```

### ADR-002: Chrono-Specific URDF

**Status:** Accepted

The original Go1 URDF has a dummy fixed root that Chrono imports as an anchor.
`models/go1/go1_chrono.urdf` removes that dummy root so the trunk is the free
body. Mesh paths are local.

### ADR-003: Stable Chrono Home Pose

**Status:** Accepted

The Menagerie home pose (`hip=0.0`, `thigh=0.9`, `calf=-1.8`) slowly sank in
this Chrono import even with zero policy action. Reward tuning was therefore
fighting a bad neutral pose.

Accepted baseline:

```text
home pose:    hip=0.0, thigh=0.7, calf=-1.4
spawn height: 0.34 m
```

This pose starts at its natural support height and holds with zero action.

### ADR-004: Zero-Overhead Home-Pose Spawn

**Status:** Accepted

`SetRootInitPose()` only initializes the root. Joints initially start at zero.
Instead of using a 500-step motor ramp, the env fixes the trunk, runs Chrono's
position assembly solver, then unfixes the trunk:

```python
self._trunk.SetFixed(True)
system.DoAssembly(1)
self._trunk.SetFixed(False)
```

This places all position motors at the home pose before the first dynamics step.

### ADR-005: Collision Whitelist

**Status:** Accepted

Enabling collision on every imported body caused solver explosions. The env
enables only the external contact envelope: trunk, hips, thighs, calves, and
feet. Rotor, camera, and marker bodies stay non-colliding.

### ADR-006: Standing Reward

**Status:** Flat-ground v1 accepted

The standing reward was rebuilt from scratch after joint observation and home
pose bugs invalidated early training runs. Terms were added one at a time and
kept only after evaluation.

Important rejected paths:

- Raising upright reward alone reduced lean but brought back more leg chatter.
- Raising angular-velocity too much made the policy too constrained and unstable.
- Raising control penalty too much reduced corrective authority and caused worse
  tipping/spinning.
- Pose penalty was not the main fix once pose error stayed near zero.

Accepted lesson: separate the failure modes. Use `xz_vel` for ground-plane drift,
`joint_vel` and `action_rate` for chatter, `ang_vel` for body motion, and `tilt`
for persistent lean.

### ADR-007: Full Rebuild On Reset

**Status:** Accepted

SCM terrain deformation cannot be cleared in place, so `reset()` rebuilds the
Chrono system. Flat and SCM terrain use the same reset path.

## Roadmap

```text
Stage 1  train_stand.py       flat terrain, fixed friction=0.8       <- active
           -> flat-ground standing v1 accepted
           -> next: reduce mild in-place shuffling
Stage 2  train_stand.py       flat terrain, friction randomized
Stage 3  train_walk.py        flat terrain walking
Stage 4  train_walk_scm.py    SCM deformable terrain fine-tuning
Stage 5  rollout collection   learned standing/walking skills
Stage 6  world model          obs/action/next_obs prediction
Stage 7  hierarchy            skill selection and planning
```

Immediate next steps:

```text
1. keep the current flat-ground standing model as v1
2. tune one smoothness term at a time to reduce mild shuffling
3. rerun fixed-friction evaluation after every reward edit
4. start friction randomization once v1 smoothness is acceptable
5. move to SCM only after flat randomized standing is stable
```

## Detailed Notes

- [docs/training_roadmap.md](docs/training_roadmap.md) - reward decisions, diagnosis log, evaluation checklist
- [docs/chrono_port_notes.md](docs/chrono_port_notes.md) - Chrono port engineering notes
- [docs/collision_debug_log.md](docs/collision_debug_log.md) - collision whitelist debug log
