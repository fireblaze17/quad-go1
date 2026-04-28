# Quad Go1

Project Chrono robotics simulation and machine learning project for a
Unitree Go1-style quadruped.

The goal is to build a clean robotics ML stack:

1. Build a stable Go1 simulation in Chrono.
2. Wrap it as a Gymnasium environment.
3. Train standing and locomotion policies.
4. Transfer from flat rigid ground to Chrono SCM deformable terrain.
5. Collect rollout data from learned skills.
6. Train a world model.
7. Add hierarchical skill selection later.

## Current Status

```text
Go1 stands in Chrono with position motors, flat/SCM terrain support,
and a stable external collision envelope.
```

Main viewer:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe view_env.py
```

`view_env.py` is the primary live test harness. It creates:

```python
Go1Env(render_mode="human", max_steps=1000, terrain=TERRAIN)
```

## Project Shape

```text
go1_env.py                 Chrono Gymnasium environment
view_env.py                main live viewer/test script
models/go1/go1_chrono.urdf Chrono-specific Go1 URDF
chrono_go1_soil.py         SCM deformable terrain milestone
mujoco/                    old MuJoCo baseline, kept for reference
docs/                      detailed decision logs and roadmap
```

## Key Decisions

### Chrono First, MuJoCo As Reference

The simulator for this project is Project Chrono. MuJoCo Menagerie is used only
as a reference for Go1 model values that were already known to work:

```text
home pose
joint limits
friction values
collision primitive placement
actuator/control conventions
```

The actual runtime path is:

```text
view_env.py -> go1_env.py -> PyChrono -> models/go1/go1_chrono.urdf
```

### Y-Up Conversion

The Go1 URDF follows ROS convention and is Z-up. This project uses Chrono in a
Y-up world. The root pose is rotated by -90 degrees around X:

```python
parser.SetRootInitPose(
    chrono.ChFramed(
        chrono.ChVector3d(0, _SPAWN_HEIGHT, 0),
        chrono.QuatFromAngleX(-math.pi / 2),
    )
)
```

This maps the URDF standing orientation into Chrono's Y-up frame.

### Chrono-Specific URDF

The original Go1 URDF had a dummy `base` link and a fixed `floating_base` joint.
Chrono treated this fixed root as the robot anchor, so the robot could hang in
the air. `models/go1/go1_chrono.urdf` removes that dummy root path so `trunk`
is the free root body.

### Position Control First

The env currently uses position actuation:

```python
parser.SetAllJointsActuationType(
    parsers.ChParserURDF.ActuationType_POSITION
)
```

Zero action targets the MuJoCo Menagerie home control pose:

```text
hip=0.0, thigh=0.9, calf=-1.8
```

Actions are normalized offsets:

```text
target = home_joint_angles + 0.25 * action
```

Tradeoff: position control is easier to stabilize for the first standing and
walking policies. Torque control can be revisited later after reward shaping and
baseline locomotion are working.

### Spawn Clearance And Target Ramp

MuJoCo Menagerie's home base height is `z=0.27`. In Chrono, that maps
conceptually to `y=0.27`, but `ChParserURDF.SetRootInitPose()` only initializes
the root body, not a full joint keyframe. At the parser's zero-joint pose, the
feet sit lower than they do in the home pose.

Current fix:

```text
spawn root at Chrono y=0.48
ramp position targets from zero to home over 1 second
```

Tradeoff: the robot starts slightly higher than the Menagerie home height, but
it avoids initial foot/ground interpenetration and motor snap.

### Collision Whitelist

`ChParserURDF` imports collision shapes but leaves collision disabled. Enabling
every non-fixed URDF body proved contacts existed, but caused solver explosions.

The stable Chrono contact envelope is a whitelist of external shell bodies:

```python
_ROBOT_COLLISION_BODIES = (
    "trunk",
    "FR_hip", "FL_hip", "RR_hip", "RL_hip",
    "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh",
    "FR_calf", "FL_calf", "RR_calf", "RL_calf",
    "FR_foot", "FL_foot", "RR_foot", "RL_foot",
)
```

Rotor and sensor marker collision bodies stay disabled. They are not part of
the external terrain-contact shell, and MuJoCo Menagerie does not expose
separate colliding rotor bodies for Go1.

Full reasoning: [docs/collision_debug_log.md](docs/collision_debug_log.md)

### Hip Collision Fix

The original Chrono URDF had one hip collision cylinder per hip body at
`y=+/-0.08`. With hip collisions enabled, that offset destabilized the
simulation.

Comparing the existing URDF hip cylinder to MuJoCo Menagerie's closest
corresponding hip primitive showed the stable placement should be `y=+/-0.045`
with the same radius `0.046`.

Rule used:

```text
Do not add new URDF collision shapes just because Menagerie has more.
Only correct values on existing URDF collision elements.
```

### Flat First, SCM Later

`Go1Env` supports both:

```python
Go1Env(terrain="flat")
Go1Env(terrain="scm")
```

Flat rigid ground is the first training stage because it is faster and easier
to stabilize. SCM deformable terrain is the later fine-tuning stage and the
main reason for moving to Chrono.

### Full Rebuild On Reset

SCM terrain deformation cannot be cleared in place. The env rebuilds the entire
Chrono system on `reset()`.

Tradeoff: this is slower than a partial reset, but it is correct for SCM and
keeps the flat/SCM reset path simple.

## Roadmap

```text
Stage 1  train_stand.py       flat terrain, fixed friction=0.8
Stage 2  train_walk.py        flat terrain, friction randomized
Stage 3  train_walk_scm.py    SCM deformable terrain fine-tuning
Stage 4  rollout collection   learned standing/walking skills
Stage 5  world model          obs/action/next_obs prediction
Stage 6  hierarchy            skill selection and planning
```

Immediate next code work:

```text
add fall termination
add standing reward
create train_stand.py for the Chrono env
```

## Detailed Notes

- [docs/chrono_port_notes.md](docs/chrono_port_notes.md)
- [docs/collision_debug_log.md](docs/collision_debug_log.md)
- [docs/training_roadmap.md](docs/training_roadmap.md)
