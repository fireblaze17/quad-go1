# Quad Go1

Robotics simulation and machine learning project for a Unitree Go1-style
quadruped using Project Chrono.

The goal is to build a clean robotics ML stack:

1. Set up a basic PyChrono simulation.
2. Build or import a quadruped model.
3. Wrap the Chrono simulator as a Gymnasium environment.
4. Train standing and locomotion policies.
5. Collect rollout data from learned skills.
6. Train a world model.
7. Add hierarchical skill selection later.

## Current Status

```text
Status: Go1 standing on SCM deformable terrain with contacts working
```

## Milestone 1: Basic Chrono Physics Demo

The first Chrono demo is implemented in:

```text
chrono_demo.py
```

It creates:

```text
Chrono physics system
Y-up gravity
fixed ground body
dynamic falling box
Bullet collision system
Irrlicht visualization window
```

The box falls onto the ground, contacts are detected, and the box settles at the
expected height.

Important Chrono setup details:

```text
ChSystemNSC
SetGravityY()
SetCollisionSystemType(Type_BULLET)
ChContactMaterialNSC
AddBody(...)
DoStepDynamics(...)
ChVisualSystemIrrlicht
```

This milestone proves that basic Chrono physics, contact, and visualization are
working before adding robots or learning code.

## Milestone 2: SCM Deformable Soil Demo

The deformable terrain demo is implemented in:

```text
chrono_go1_soil.py
```

It creates:

```text
Chrono SMC system
SCM deformable terrain
soil parameter setup
Irrlicht visualization
simple falling test box
```

The SCM terrain uses Chrono's vehicle terrain module:

```python
pychrono.vehicle.SCMTerrain
```

Since the rest of the project uses Y-up coordinates, the SCM terrain reference
frame is rotated to match the simulation frame.

The current soil parameters come from Chrono's own deformable-soil demo and are
used as a starter terrain, not as a final Mars/regolith model.

This milestone proves that Chrono's deformable soil terrain can be created,
visualized, and stepped before adding the Go1 robot.

## Milestone 3: Go1 URDF Parser Check

The Unitree Go1 URDF and mesh assets were added under:

```text
models/go1/
```

The URDF originally referenced ROS package paths such as:

```text
package://go1_description/meshes/trunk.dae
```

Those paths were converted to local mesh paths:

```text
meshes/trunk.dae
```

The parser test script is:

```text
load_go1_urdf.py
```

Current goal of that script:

```text
1. Load the Go1 URDF with ChParserURDF.
2. Print URDF bodies and joints.
3. Populate a Chrono system.
4. Print the Chrono bodies and joints created by the parser.
```

This is the first step toward importing the Go1 model into the Chrono
simulation.

## Milestone 4: Chrono-Specific Go1 URDF

The original Go1 URDF contained a dummy root link:

```text
base
```

and a joint named:

```text
floating_base
```

but that joint was actually fixed:

```xml
<joint name="floating_base" type="fixed">
```

Chrono imported `base` as the fixed root body, which caused the robot to hang in
the air instead of falling under gravity.

To fix this cleanly, a Chrono-specific URDF was created:

```text
models/go1/go1_chrono.urdf
```

This version removes the dummy `base` link and the fixed `floating_base` joint,
making `trunk` the free root body.

Current import check:

```text
root body: trunk
root fixed: False
```

The robot now falls under gravity in Chrono. The next issue to solve is contact
between the imported Go1 collision geometry and the SCM deformable terrain.

## Milestone 5: Go1 Contact with SCM Terrain Fixed

### Problem

After loading the Go1 via `ChParserURDF` and running `inspect_go1_contacts.py`,
every body reported `collision_enabled= False` and contacts stayed at zero
throughout the simulation:

```text
trunk collision_enabled= False mass= 5.204 pos_y= 0.452
FR_foot collision_enabled= False mass= 0.06 pos_y= 0.3232
...

time= 0.002 trunk_y= 0.452 contacts= 0
time= 0.402 trunk_y= -0.3438 contacts= 0
time= 0.802 trunk_y= -2.7076 contacts= 0
```

The robot fell straight through the SCM terrain indefinitely.

### Root Cause

`ChParserURDF` builds collision shapes from the URDF `<collision>` tags but
leaves collision **disabled** on every body it creates. Collision must be
activated explicitly after `PopulateSystem()`.

### Fix

In `load_go1()` inside `view_go1_urdf.py`, after `PopulateSystem()`, iterate
every non-fixed body and call `EnableCollision(True)`:

```python
for body in system.GetBodies():
    if not body.IsFixed():
        body.EnableCollision(True)
```

This is now done centrally in `load_go1()` so all scripts that import it get
the fix automatically.

### Result

All bodies now show `collision_enabled= True` and contacts register when the
feet reach the terrain surface. The Go1 lands and rests on the SCM soil instead
of falling through.

## Milestone 6: Full Parser Configuration

After getting contacts working, the parser setup in `load_go1()` was hardened
with correct physics and actuation settings.

### Mesh collision type

Changed from convex hull to triangle mesh:

```python
parser.SetAllBodiesMeshCollisionType(
    parsers.ChParserURDF.MeshCollisionType_TRIANGLE_MESH
)
```

Convex hull loses concave surface detail, which causes limb interpenetration on
Go1's curved leg geometry. Triangle mesh preserves the exact shape.

### Force actuation

All joints are set to force (torque) actuation to match the real hardware:

```python
parser.SetAllJointsActuationType(
    parsers.ChParserURDF.ActuationType_FORCE
)
```

The real Unitree Go1 uses `EffortJointInterface` (torque control). With no
torque function applied yet the joints are free under gravity, so the robot
collapses/flails in the viewer. This is expected — torques will be supplied by
the Gymnasium env action step.

### Contact materials

Contact material values were taken from MuJoCo Menagerie `unitree_go1/go1.xml`.
`ChContactMaterialData` is used (not `ChContactMaterialSMC`, which raises a
`TypeError`):

```python
mat = chrono.ChContactMaterialData()
mat.mu = 0.6   # sliding friction — all body surfaces
mat.cr = 0.0   # inelastic (matches MuJoCo default)
parser.SetDefaultContactMaterial(mat)
```

Foot surfaces use higher friction (matching MuJoCo's foot geom value of 0.8),
applied per-body after `PopulateSystem()`:

```python
foot_mat = chrono.ChContactMaterialData()
foot_mat.mu = 0.8
foot_mat.cr = 0.0
for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
    parser.SetBodyContactMaterial(name, foot_mat)
```

## Milestone 7: URDF Cleanup — Zero Inertia Warnings Removed

The original `go1_chrono.urdf` contained 7 empty links that produced
zero-inertia warnings on every run:

```text
camera_optical_face, camera_optical_chin, camera_optical_left,
camera_optical_right, camera_optical_rearDown,
camera_laserscan_link_left, camera_laserscan_link_right
```

These are ROS sensor frame markers with no mass, inertia, or collision. Chrono
cannot simulate massless non-fixed bodies, so it printed warnings and skipped
them. The links and their connecting fixed joints were removed from
`models/go1/go1_chrono.urdf`. The simulation now loads cleanly with no
warnings.

### Problems encountered during URDF cleanup

- **Problem:** Removing the dummy `base` root link (Milestone 4) caused the
  parser to treat the first real link as a fixed ground body.
  **Fix:** Also remove the associated `floating_base` fixed joint, making
  `trunk` the free-floating root.

- **Problem:** Joint dynamics show `damping="0.0" friction="0.0"` (official
  Unitree values). This is intentional — damping is handled by the motor
  controllers on real hardware and will be replicated in the Gym env.

## Next Steps

1. Add reward shaping and fall termination to `go1_env.py` `step()`.
2. Train a standing policy (`train_stand.py`).
3. Train a walking policy (`train_walk.py`).
4. Collect rollout data from learned skills.
5. Train a world model.
6. Add hierarchical skill selection.

---

## Milestone 8: Gymnasium Environment Wrapper (`go1_env.py`)

The full Chrono simulation is now wrapped as a standard `gymnasium.Env` in:

```text
go1_env.py
```

### Observation space (37-dim float32)

```text
trunk position       (3)   x y z
trunk quaternion     (4)   w x y z  (Chrono: e0=w, e1=x, e2=y, e3=z)
trunk linear vel     (3)
trunk angular vel    (3)
joint angles         (12)
joint velocities     (12)
```

### Action space (12-dim float32, in [-1, 1])

Normalised torques scaled by per-joint limits at apply time:

```text
hip=23.7 N·m, thigh=23.7 N·m, calf=35.55 N·m  (official Unitree values)
```

Joint order: FR, FL, RR, RL — each hip/thigh/calf.

### Problems encountered and how they were fixed

**1. `ChParserURDF.GetChMotor()` returns `ChLinkMotor`, not `ChLinkMotorRotationTorque`**

The base class `ChLinkMotor` does not expose `SetTorqueFunction()`. Calling it
raised `AttributeError`. The correct method is `SetMotorFunction()`, which
accepts a `ChFunctionConst`. The constant value is updated in-place each step
via `SetConstant()`, avoiding object allocation per step.

**2. Joint angle reading — `GetMotorAngle()` not accessible**

`GetMotorAngle()` and `GetMotorAngleDt()` are defined on the derived
`ChLinkMotorRotationAngle` class but are not accessible through the
`ChLinkMotor` base pointer returned by `GetChMotor()`. Joint angles are instead
computed from the relative rotation of the two body frames that each motor
connects:

```python
q_rel = q1.GetInverse() * q2
angle = q_rel.GetRotVec().z   # Z-component = rotation around joint axis
```

Joint velocities use the relative angular velocity in body1's local frame.

**3. `ChVisualSystemIrrlicht` has no `DetachSystem()`**

On `reset()` the simulation is fully rebuilt. Calling `AttachSystem()` on an
already-initialised `ChVisualSystemIrrlicht` crashes the Irrlicht device. The
fix is to always create a fresh `ChVisualSystemIrrlicht` on every `_build_sim()`
call, setting `self._vis = None` before creating the new one.

**4. `SetBodyContactMaterial()` must be called before `PopulateSystem()`**

Initially the foot material overrides were set after `PopulateSystem()`, which
printed a warning and had no effect. Moved to before `PopulateSystem()`.

**5. SCMTerrain accumulates deformation state — cannot be cleared**

There is no API to reset SCM terrain deformation in place. The entire Chrono
system must be rebuilt on every `reset()`. This is expensive but unavoidable
for SCM. For the flat terrain mode it is a minor cost.

**6. Full system rebuild on reset crashes if visualizer is reused**

Calling `AttachSystem()` a second time on the same initialised
`ChVisualSystemIrrlicht` crashes Irrlicht. Fixed by creating a new visualizer
every `_build_sim()` (see problem 3 above).

### Tradeoffs

**Full rebuild on reset vs. partial reset**

A full rebuild is required for SCM because the terrain deformation cannot be
cleared. For flat ground this is unnecessary overhead, but keeping one code path
for both terrain types is simpler and the rebuild cost on flat is negligible
(~10 ms). Tradeoff accepted: simplicity over micro-optimisation.

**`ChFunctionConst` pre-allocation**

Torque functions are pre-allocated once and updated via `SetConstant()` rather
than creating new `ChFunctionConst` objects each step. This avoids repeated
small allocations during the inner training loop. Tradeoff: slightly more state
to manage in `_build_sim()`.

**`reward = 0.0` placeholder**

Reward shaping was deliberately deferred. The env interface and physics are
validated first (zero-torque passive fall with `view_env.py`). Training with a
zero reward would teach nothing, but the env itself runs correctly. The
`_TERM_HEIGHT` constant exists but `terminated = False` always until reward
shaping is added.

---

## Milestone 9: Terrain Switching and Ground Materials

### Switchable terrain

`Go1Env` accepts a `terrain=` parameter:

```python
Go1Env(terrain="flat")  # default — rigid, fast
Go1Env(terrain="scm")   # deformable Bekker-SCM soil, ~5× slower
```

`view_env.py` exposes this as a single `TERRAIN` constant at the top of the
file.

### Why flat first

The standard approach in legged locomotion RL papers (ANYmal ETH 2022, RMA 2021,
Unitree Go1/Go2 work) is to train on flat rigid ground first, then apply
curriculum. Flat is ~5× faster per step, numerically stable, and allows the
policy to learn basic locomotion before fighting terrain variability. SCM is
deferred to a fine-tuning stage.

### Ground contact material — matched to MuJoCo Menagerie

Ground and foot materials are matched to the values from
`mujoco_menagerie/unitree_go1/go1.xml` so both simulators start from the same
physical baseline:

```text
MuJoCo:
  floor friction (scene.xml): default engine values — sliding=1.0, rolling=0.0001
  foot  friction (go1.xml):   friction="0.8 0.02 0.01"  (sliding, torsional, rolling)
  effective contact = min(floor, foot) = sliding=0.8

Chrono composite rule: average of the two materials
  ground SetFriction(0.8) + foot SetFriction(0.8) → effective = 0.8  ✓ matches MuJoCo
  ground SetRollingFriction(0.0001)  — MuJoCo floor rolling default
```

**Why average vs. minimum matters:**
MuJoCo uses the minimum of the two geom friction values at a contact pair.
Chrono uses the average by default. To get the same effective sliding friction
(0.8), both the ground and foot materials must be set to 0.8 in Chrono.
Setting the ground to 1.0 (the MuJoCo floor default) would give an effective of
0.9 in Chrono — slightly higher than MuJoCo. Matching both to 0.8 aligns the
simulators.

### Visualisation colours

Ground and robot body colours are set via `ChVisualModel` shape iteration after
`PopulateSystem()`:

```text
Ground:  ChColor(0.05, 0.05, 0.05)  — near-black
Robot:   ChColor(0.2, 0.45, 0.85)   — steel blue
```

Collision shapes (capsules from `EnableCollisionVisualization()`) are also
tinted since they share the same visual model.

### SCM soil parameters (for reference)

The SCM preset is dry loose sand — Chrono's own demo defaults:

```text
Bekker Kphi:        2e5  Pa/m^(n+1)   frictional modulus
Bekker Kc:          0                  no cohesive modulus (dry sand)
Bekker n:           1.1                sinkage exponent
Mohr cohesion:      0    Pa            dry sand = 0
Mohr friction:      30°
Janosi shear K:     0.01 m
Elastic stiffness:  4e7  Pa/m
Damping:            3e4  Pa·s/m
```

Wet clay would have higher Kc and cohesion. Hard packed soil would have much
higher Kphi and elastic stiffness. The dry sand preset is a reasonable starting
point for outdoor locomotion before parameter identification.

---

## Training Roadmap

```text
Stage 1  train_stand.py       flat terrain, fixed friction=0.8    standing policy
Stage 2  train_walk.py        flat terrain, friction randomised    walking policy
Stage 3  train_walk_scm.py    SCM deformable terrain, fine-tune   soft terrain transfer
```

Stage 3 using Chrono SCM is novel — no mainstream RL simulator (Isaac Gym,
MuJoCo) has proper deformable terrain physics, so this is unpublished territory
and potentially publishable as a contribution.

Before any training can start, `step()` needs:
- A standing/walking reward function (trunk height, upright score, motor penalty)
- Fall termination (`terminated = trunk_y < _TERM_HEIGHT`)
