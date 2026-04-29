# Chrono Port Notes

Architecture Decision Records for the Project Chrono Go1 port. Each section is
the context that forced a decision, what was decided, and what it costs.
Git history records what was tried. These docs record only the current rational state.

---

## ADR-001: Y-Up World

**Status:** Accepted

**Context:** The Go1 URDF follows ROS Z-up convention. Chrono supports both
conventions. SCM terrain has a native Z-up frame. The project needed to pick one.

**Decision:** Y-up world (`system.SetGravityY()`). Root pose rotated −90° about X on import:

```python
chrono.QuatFromAngleX(-math.pi / 2)
```

SCM terrain reference frame also rotated −90° about X to match.

**Consequences:**
- All world-space vectors use Y as up: `trunk_y` = height.
- Trunk upright alignment: `Rotate(ChVector3d(0,0,1)).y` — local Z dotted with world Y.
- Thigh/calf joint axes: URDF Y → Chrono −Z after spawn rotation (see ADR-008).

---

## ADR-002: Chrono-Specific URDF

**Status:** Accepted

**Context:** The source Go1 URDF has a dummy `base` link and a fixed `floating_base`
joint. Chrono imported the fixed root as an anchor — the robot hung in the air
rather than falling under gravity. ROS package mesh paths also needed to be local.

**Decision:** `models/go1/go1_chrono.urdf` — removes the dummy root so `trunk` is
the free root body. Mesh paths converted from `package://go1_description/meshes/`
to local `meshes/`.

**Consequences:**
- URDF diverges from upstream source — must be maintained separately.
- Rule: never add new URDF collision elements just because Menagerie has more.
  Only correct values on existing elements (see ADR-005).

---

## ADR-003: Position Control

**Status:** Accepted

**Context:** First policy needs a stable actuation baseline. Torque control requires
tuned PD gains and is harder to stabilize initially.

**Decision:** `ActuationType_POSITION` for all joints. Zero action holds the MuJoCo
Menagerie home pose (`hip=0.0, thigh=0.9, calf=−1.8`). Actions are normalized
offsets: `target = home + 0.25 × action`.

**Consequences:**
- Easier to stabilize for first standing and walking policies.
- Torque control is a later consideration after locomotion baselines work.

---

## ADR-004: Collision Whitelist

**Status:** Accepted

**Context:** `ChParserURDF` imports collision shapes with collision disabled. Enabling
all non-fixed bodies caused solver explosions (robot launches, scuttles, then
explodes) even with motors disabled. See [collision_debug_log.md](collision_debug_log.md)
for the staged debug process.

**Decision:** Disable all collision after import, then enable only the external
contact envelope:

```python
_ROBOT_COLLISION_BODIES = (
    "trunk",
    "FR_hip", "FL_hip", "RR_hip", "RL_hip",
    "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh",
    "FR_calf", "FL_calf", "RR_calf", "RL_calf",
    "FR_foot", "FL_foot", "RR_foot", "RL_foot",
)
```

Rotor, camera, and sensor marker bodies stay disabled.

**Consequences:**
- Matches MuJoCo Menagerie — Menagerie exposes no separate rotor collision bodies.
  Actuator effects are represented through joint parameters (damping, armature, etc.).
- URDF rotor bodies may still contribute mass/inertia but not terrain contact.

---

## ADR-005: Hip Collision Origin

**Status:** Accepted

**Context:** The original URDF hip collision cylinder was at `y=+/−0.08`. Enabling
hip collision destabilised the simulation.

**Decision:** Correct the existing element to match MuJoCo Menagerie reference values:

```xml
<origin rpy="1.5707963267948966 0 0" xyz="0 +/-0.045 0"/>
<cylinder length="0.04" radius="0.046"/>
```

**Consequences:**
- _Rejected: adding extra Menagerie hip primitives_ — tested and destabilised Chrono.
  One cylinder per hip is sufficient.
- Rule: correct values on existing URDF elements only. Do not add new shapes.

---

## ADR-006: Contact Materials

**Status:** Accepted

**Context:** Contact material values needed for stable simulation on flat and SCM terrain.

**Decision:** Reference values from MuJoCo Menagerie:

```text
default body friction:  0.6
foot friction:          0.8
restitution:            0.0
```

Foot material override set before `PopulateSystem()`.

**Consequences:**
- Floor friction is domain-randomized per episode (`friction_range`) as the first
  domain-randomization knob. Body friction stays fixed.

---

## ADR-007: Full Rebuild On Reset

**Status:** Accepted

**Context:** SCM terrain deformation cannot be cleared in place. `ChVisualSystemIrrlicht`
cannot be safely reattached to a rebuilt system after initialization.

**Decision:** `reset()` tears down and rebuilds the entire Chrono system from scratch.
A fresh visualizer is created each build when `render_mode="human"`.

**Consequences:**
- Slower than a partial reset.
- Correct for both flat and SCM terrain — single reset path, no special cases.

---

## ADR-008: Joint Axis Reading

**Status:** Accepted

**Context:** `CastToChLinkMotorRotation(motor).GetMotorAngle()` is not exposed in
this PyChrono build. Joint angles must be computed from body-frame rotation vectors.

**Decision:** Per-joint axis map and sign correction applied to `GetRotVec()`:

```python
_JOINT_AXES      = np.array([0,2,2, 0,2,2, 0,2,2, 0,2,2], dtype=np.int32)
_JOINT_AXIS_SIGN = np.where(_JOINT_AXES == 0, 1.0, -1.0)
```

Geometric derivation:

- **Hip:** URDF `axis="1 0 0"` (X). After −90° spawn rotation, still Chrono X.
  `GetRotVec().x = θ`. Sign = +1.
- **Thigh/calf:** URDF `axis="0 1 0"` (Y). After −90° spawn rotation, Y → Chrono −Z.
  Rotation of angle θ about −Z gives `GetRotVec().z = −θ`.
  At home (thigh θ = 0.9 rad): `GetRotVec().z = −0.9`.
  `sign=−1`: reading = `−1 × (−0.9) = +0.9` ✓ matches `_HOME_JOINT_ANGLES[thigh]`
  `sign=+1`: reading = `+1 × (−0.9) = −0.9` ✗ → pose_error = 3.24 per joint → 0% survival

**Consequences:**
- Full retrain required when axis constants change — hip observation values shift from ~0 to real angles.
- _Rejected: sign=+1_ — see derivation above. Caused total pose_penalty ≈ 6.5/step → −6500/episode.

---

## ADR-009: Observation Space

**Status:** Accepted

**Context:** Gymnasium env needs a fixed observation vector matching the MuJoCo baseline.

**Decision:** 37-dim observation:

```text
trunk position       3   (pos.x, pos.y, pos.z)
trunk quaternion     4   (e0, e1, e2, e3)
trunk linear vel     3
trunk angular vel    3
joint angles         12  (FR/FL/RR/RL × hip/thigh/calf)
joint velocities     12
total                37
```

**Consequences:**
- Joint order must stay synchronized with `_JOINT_NAMES`, `_JOINT_AXES`,
  `_HOME_JOINT_ANGLES`, `_JOINT_LOW/HIGH`.

---

## ADR-010: Home-Pose Spawn — `DoAssembly(POSITION)`

**Status:** Accepted — see [training_roadmap.md ADR-007](training_roadmap.md)

**Context:** `ChParserURDF.SetRootInitPose()` initialises only the root body. All
joint angles start at 0, placing feet ~0.5 m below the trunk. The original fix
was a 500-step motor ramp — wasted training signal. A warm-up loop (50–200 steps)
worked but added ~5% per-episode overhead.

**Decision:** After setting motors to home angles, run Chrono's kinematic assembly
solver before the first `DoStepDynamics()`:

```python
self._trunk.SetFixed(True)
system.DoAssembly(1)   # pure constraint satisfaction — no forces, no time integration
self._trunk.SetFixed(False)
```

**Consequences:**
- Zero compute overhead. Robot spawns at home pose, drops ~0.08 m to the floor, settles.
- _Rejected: motor ramp_ — wasted 500 training steps per episode.
- _Rejected: warm-up loop_ — ~5% overhead rejected as unnecessary.
- _Inspiration:_ [harryzhang1018](https://github.com/harryzhang1018) /
  [SBEL multi-terrain RL](https://github.com/uwsbel/sbel-reproducibility/tree/master/2025/multi-terrain-RL)
  (UW-Madison, 2025) — calls `actuate(home)` before any physics steps.
  Our `DoAssembly` achieves the same correctly without a warm-up loop.

---

## SCM Terrain Parameters

Current parameters from Chrono's deformable-soil demo — starter preset only:

```text
Bekker Kphi:        2e5
Bekker Kc:          0
Bekker n:           1.1
Mohr cohesion:      0
Mohr friction:      30 deg
Janosi shear K:     0.01
Elastic stiffness:  4e7
Damping:            3e4
```

These will need tuning against real soil data before SCM training (Stage 4).

## Basic Chrono Physics

Early falling-box smoke tests verified the basic Chrono stack:

```text
ChSystemNSC
Y-up gravity
Bullet collision system
fixed ground box
falling dynamic box
Irrlicht visualization
```

Those tests proved that Chrono physics, contact, and visualization worked before
adding a robot. The standalone demo script was removed after the Go1 env and SCM
terrain examples became the maintained test paths.

## SCM Deformable Terrain

`chrono_go1_soil.py` verifies Chrono's SCM terrain module:

```python
pychrono.vehicle.SCMTerrain
```

The rest of the project uses Y-up coordinates, so the SCM terrain reference
frame is rotated by -90 degrees about X, matching the robot import frame.

The current SCM parameters come from Chrono's deformable-soil demo and are only
a starter preset:

```text
Bekker Kphi:        2e5
Bekker Kc:          0
Bekker n:           1.1
Mohr cohesion:      0
Mohr friction:      30 deg
Janosi shear K:     0.01
Elastic stiffness:  4e7
Damping:            3e4
```

These will need tuning against real soil data before SCM training (Stage 4).

