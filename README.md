# Quad Go1

Project Chrono robotics simulation and ML project for a Unitree Go1-style quadruped.

## Goal

1. Build a stable Go1 simulation in Chrono.
2. Wrap it as a Gymnasium environment.
3. Train standing then locomotion policies.
4. Transfer to Chrono SCM deformable terrain.
5. Collect rollouts, train a world model, add hierarchical skill selection.

## Current Status

```text
Stage 1 — standing policy, flat terrain, fixed friction=0.8

Active reward:
  alive_bonus(+1.0) + upright_score(×1.0) − pose_penalty(×0.15) − control_penalty(×0.01) − ang_vel_penalty(×0.05)

Solved so far:
  ✓ Chrono simulation + Y-up world
  ✓ Chrono-specific Go1 URDF (trunk as free root)
  ✓ Hip joint axis bug (hips now visible to pose penalty)
  ✓ height_score → terrain-agnostic alive_bonus
  ✓ Slow-sink fixed (pose_weight raised 0.1 → 0.15)
  ✓ Motor ramp removed (DoAssembly drives joints to home before first physics step — zero overhead)

Pending:
  ✗ Retrain with pose_weight=0.15 + no-ramp spawn
  ✗ position_penalty (0.5 × Σxy²) — one term at a time after retrain
  ✗ xy_vel_penalty (0.1 × Σxy_vel²)
```

## Project Shape

```text
go1_env.py                 Chrono Gymnasium environment
view_env.py                live test harness
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
# View environment (no policy)
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe view_env.py

# Train standing policy
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000

# Evaluate
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe evaluate_stand.py runs/stand/final_model.zip

# View trained policy
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe view_stand_policy.py runs/stand/final_model.zip
```

## Technical Decisions

Each section: what forced the decision, what was chosen, what it costs.
Full ADRs: [docs/chrono_port_notes.md](docs/chrono_port_notes.md) · [docs/training_roadmap.md](docs/training_roadmap.md) · [docs/collision_debug_log.md](docs/collision_debug_log.md)

---

### ADR-001: Y-Up World + Position Control

**Status:** Accepted

**Context:** Go1 URDF is ROS Z-up. Chrono supports both. SCM terrain is Y-up native.

**Decision:** Y-up world (`SetGravityY`). Root pose rotated −90° about X on import.
All joints use `ActuationType_POSITION`. Zero action = MuJoCo Menagerie home pose (`hip=0.0, thigh=0.9, calf=−1.8`). Actions are normalized offsets: `target = home + 0.25 × action`.

---

### ADR-002: Chrono-Specific URDF

**Status:** Accepted

**Context:** Original Go1 URDF has a dummy `base` link and fixed `floating_base` joint. Chrono treats the fixed root as an anchor — robot hangs in the air.

**Decision:** `models/go1/go1_chrono.urdf` — removes dummy root so `trunk` is the free body. Mesh paths converted from `package://go1_description/meshes/` to local paths.

---

### ADR-003: Standing Reward

**Status:** Accepted — retrain pending

**Context:** One term added per training run, evaluated before adding the next.

**Decision:**

```python
reward = alive_bonus + upright_score - pose_penalty - control_penalty - ang_vel_penalty
```

| Term | Weight | Problem solved |
|---|---|---|
| `alive_bonus` | +1.0/step | terrain-agnostic positive signal (no Y=0 assumption) |
| `upright_score` | ×1.0 | robot pitched forward |
| `pose_penalty` | ×0.15 | legs folded under trunk; slow height sink |
| `control_penalty` | ×0.01 | policy saturated joint targets |
| `ang_vel_penalty` | ×0.05 | trunk spinning and rotational jitter |

Pending (one at a time): `position_penalty` (×0.5), `xy_vel_penalty` (×0.1).

---

### ADR-004: Zero-Overhead Home-Pose Spawn

**Status:** Accepted

**Context:** `SetRootInitPose()` initialises only the root body — joints start at 0 rad. Setting motors to home angles before the first step causes a constraint-snap impulse that launches the robot.

**Decision:** Chrono's built-in kinematic assembly solver:

```python
self._trunk.SetFixed(True)
system.DoAssembly(1)   # AssemblyAnalysis.POSITION — pure constraint satisfaction
self._trunk.SetFixed(False)
```

Zero compute overhead. Robot spawns at home pose, drops ~0.08 m to the floor, settles.

**Consequences:**
- _Rejected: motor ramp_ — wasted 500 training steps per episode.
- _Rejected: warm-up loop_ — ~5% per-episode overhead.
- _Inspired by_ [harryzhang1018](https://github.com/harryzhang1018) /
  [SBEL multi-terrain RL](https://github.com/uwsbel/sbel-reproducibility/tree/master/2025/multi-terrain-RL)
  (UW-Madison SBEL, 2025).

---

### ADR-005: Collision Whitelist

**Status:** Accepted

**Context:** Enabling all non-fixed bodies caused solver explosions even with motors disabled.

**Decision:** Disable all collision after import, then enable only the external contact envelope (trunk, hips, thighs, calves, feet). Rotor and sensor bodies stay disabled — matches MuJoCo Menagerie, which has no separate rotor collision bodies.

---

### ADR-006: Full Rebuild On Reset

**Status:** Accepted

**Context:** SCM terrain deformation cannot be cleared in place.

**Decision:** `reset()` tears down and rebuilds the entire Chrono system. `terrain="flat"` and `terrain="scm"` both use the same reset path.

## Roadmap

```text
Stage 1  train_stand.py       flat terrain, fixed friction=0.8       ← active
           ↳ position_penalty (0.5×Σxy²)
           ↳ xy_vel_penalty (0.1×Σxy_vel²)
Stage 2  train_stand.py       flat terrain, friction randomized (0.6–1.0)
Stage 3  train_walk.py        flat terrain walking
Stage 4  train_walk_scm.py    SCM deformable terrain fine-tuning
Stage 5  rollout collection   learned standing/walking skills
Stage 6  world model          obs/action/next_obs prediction
Stage 7  hierarchy            skill selection and planning
```

Immediate next steps:

```text
1. retrain — pose_weight=0.15, no-ramp spawn, _SPAWN_HEIGHT=0.35
2. evaluate — survival_rate=1.0, mean_reward ≈ 900–1000, trunk_y stable from step 0
3. add position_penalty (0.5 × Σxy²), retrain, evaluate
4. add xy_vel_penalty (0.1 × Σxy_vel²), retrain, evaluate
5. friction randomization → Stage 2
```

## Detailed Notes

- [docs/training_roadmap.md](docs/training_roadmap.md) — reward decisions, diagnosis log, evaluation checklist
- [docs/chrono_port_notes.md](docs/chrono_port_notes.md) — Chrono port engineering notes
- [docs/collision_debug_log.md](docs/collision_debug_log.md) — collision whitelist debug log
