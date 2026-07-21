# Chrono Port Notes

Architecture Decision Records for the Project Chrono Go1 port. Each section is
the context that forced a decision, what was decided, and what it costs.

Git history records what was tried. These notes record the current rational
state.

Current standing-policy experiments are documented in
[training_roadmap.md](training_roadmap.md),
[reproduction_ladder.md](reproduction_ladder.md), and
[experiments/fixed_friction_standing.md](experiments/fixed_friction_standing.md).
This file stays focused on Chrono import, contact, reset, and physics
decisions.

---

## ADR-000: WSL Runtime For PyChrono

**Status:** Accepted

**Context:** The project originally ran PyChrono from a native Windows conda
environment. That path became unreliable when Windows Smart App Control blocked
unsigned Chrono binary extensions during import. The observed blocked files
included `Chrono_vehicle.dll` and `pychrono/_parsers.pyd`.

**Decision:** Use WSL Ubuntu as the active runtime for this repo. Activate the
WSL conda environment before running project commands:

```bash
conda activate chrono-go1
```

After activation, use `python`. On Ankus's WSL machine, the equivalent explicit
interpreter is `/home/ankus/miniforge3/envs/chrono-go1/bin/python`.

WSLg is the visualization path for Irrlicht viewers. If viewer windows or GUI
state become inconsistent, close the GUI windows and reset WSL from Windows
PowerShell with:

```powershell
wsl --shutdown
```

**Consequences:**

- Reproduction no longer depends on local Windows application-control policy.
- Native Windows repo copies should be treated as old backups unless explicitly
  synchronized.
- If another user sees blocked Chrono DLL or `_parsers.pyd` import errors, the
  recommended project path is to move to WSL rather than keep reinstalling
  packages inside native Windows.
- Viewer behavior depends on WSLg, so GUI issues may need a WSL restart even
  when headless evaluation is fine.

---

## ADR-001: Y-Up World

**Status:** Accepted

**Context:** The Go1 URDF follows ROS Z-up convention. Chrono supports multiple
world conventions. The project needed one convention before building flat and
SCM environments.

**Decision:** Use a Y-up Chrono world. The imported robot root is rotated -90
degrees about X:

```python
chrono.QuatFromAngleX(-math.pi / 2)
```

SCM terrain reference frames are also rotated -90 degrees about X to match.

**Consequences:**

- World height is `trunk_y`.
- Ground-plane velocity is X/Z, not X/Y.
- Trunk upright alignment is the trunk local Z axis dotted with world Y.
- URDF thigh/calf Y axes map to Chrono -Z after the spawn rotation.

---

## ADR-002: Chrono-Specific URDF

**Status:** Accepted

**Context:** The source Go1 URDF has a dummy `base` link and a fixed
`floating_base` joint. Chrono imported that fixed root as an anchor, so the
robot hung in the air. ROS package mesh paths also needed to be local.

**Decision:** Maintain `models/go1/go1_chrono.urdf`. It removes the dummy root
so `trunk` is the free body and converts mesh paths to local files.

**Consequences:**

- The Chrono URDF intentionally diverges from upstream.
- Keep changes minimal. Correct existing values when needed; do not add new
  collision primitives just because another simulator has them.

---

## ADR-003: Position Control And Action Scale

**Status:** Superseded for runtime actuation by ADR-013

**Context:** The first standing and walking policies need a stable actuation
baseline. Torque control requires tuned PD gains and is harder to stabilize
initially.

**Decision:** Use `ActuationType_POSITION` for all joints. Zero action held the
accepted Chrono standing pose. Actions are normalized offsets:

```python
target = home + 0.20 * action
```

The scale started larger, but `0.20` gave calmer stance corrections after the
contact and foot-support fixes.

**Consequences:**

- Easier to stabilize for the standing baseline.
- Chrono position motors remain the active runtime actuator for the current
  V3.1 clean-standing baseline.
- Walking may need the scale revisited if 0.20 limits step length.
- The action-scale decision should be revisited when locomotion policies are
  introduced.

---

## ADR-004: MaGIC-Style Rigid Contact Solver

**Status:** Accepted

**Context:** The standing policy became sensitive to foot slip, contact jitter,
and non-foot link contact. The UW-Madison/SBEL MaGIC 2025 Chrono Go2 tutorial
uses a more deliberate rigid-contact setup than the early default configuration.

**Decision:** Adopt the relevant MaGIC-style rigid-contact settings:

```python
system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
system.GetSolver().AsIterative().SetMaxIterations(60)
```

Flat-ground material:

```text
friction:     per-episode friction range
restitution: 0.1
Kn:          2e5
Gn:          60.0
```

Foot material override after URDF import:

```text
friction:     2.0
restitution: 0.01
Gn:          60.0
```

Foot material is applied directly to each foot collision model with
`GetCollisionModel().SetAllShapesMaterial(...)`.

Chrono's SMC material composition uses the minimum friction of the two contact
materials. Foot friction was historically `0.9`, which capped all higher ground
friction values at effective `0.9`. It is now intentionally set to `2.0` so
ground slices through `mu=1.2` are not capped by the foot material. This is a
cap-removal modeling choice, not a measured Unitree Go1 foot material.

**Consequences and tradeoffs:**

- More stable and intentional for rigid contact.
- Ground friction can still be randomized through `friction_range`.
- Effective friction is `min(ground_friction, foot_friction)` in the active SMC
  setup.
- This is not a full clone of the MaGIC tutorial. This project uses Go1, a
  different reward stack, and standing-first training.
- Contact-force rewards and diagnostics should be rechecked when moving to SCM.

---

## ADR-005: Collision Whitelist

**Status:** Accepted

**Context:** `ChParserURDF` imports collision shapes with collision disabled.
Enabling all non-fixed bodies caused solver instability. The first stable
whitelist kept the full external envelope: trunk, hips, thighs, calves, and
feet. That was good for zero-action stability but bad for policy learning.

Viewer diagnostics later showed huge non-foot loads:

```text
calf_load/thigh_load thousands of newtons
leg_nonfoot_load around 12000-16000 N
foot_load around 150-225 N
```

The policy could lean on calf/thigh collision geometry instead of learning clean
foot support.

**Decision:** Disable all collision after import, then enable only:

```python
_ROBOT_COLLISION_BODIES = (
    "trunk",
    "FR_foot", "FL_foot", "RR_foot", "RL_foot",
)
```

Hips, thighs, calves, rotors, camera bodies, and sensor marker bodies remain
non-colliding.

**Why this worked:** it removes hidden support modes. The robot must stand
through its feet. After adding a four-foot contact reward, the policy stopped
using the raised/unloaded foot stance.

**Tradeoffs:**

- Less literal for falls and leg scrapes, because thigh/calf terrain contact is
  ignored.
- Better for standing and walking policy learning, because normal locomotion
  should be foot-ground contact.
- Trunk collision remains enabled so falls still interact with terrain.
- The viewer still reports calf/thigh/hip contact load so regressions are
  obvious.

---

## ADR-006: Hip Collision Origin

**Status:** Historical fix, currently not active in training contact

**Context:** With hips added to the old whitelist, the simulation destabilized.
The original hip cylinder used `y=+/-0.08`, which placed the collision geometry
too far outboard.

**Decision:** The URDF hip collision origin was corrected to the Menagerie-like
value:

```xml
<origin rpy="1.5707963267948966 0 0" xyz="0 +/-0.045 0"/>
<cylinder length="0.04" radius="0.046"/>
```

**Current consequence:** hip collision is now disabled during training, so this
is no longer part of normal standing support. Keep the correction in the URDF
because it is still the better geometry if hip collision is ever re-enabled for
fall-damage or scrape studies.

---

## ADR-007: Full Rebuild On Reset

**Status:** Accepted

**Context:** SCM terrain deformation cannot be cleared in place. Irrlicht
visualization also cannot be safely reattached to a rebuilt system after
initialization.

**Decision:** `reset()` tears down and rebuilds the entire Chrono system. A fresh
visualizer is created each build when `render_mode="human"`.

**Consequences:**

- Slower than a partial reset.
- Correct for both flat and SCM terrain.
- One reset path avoids flat/SCM drift.

---

## ADR-008: Joint Angle Reading

**Status:** Accepted

**Context:** `CastToChLinkMotorRotation(motor).GetMotorAngle()` is not exposed in
this PyChrono build. The first workaround computed joint angles from linked
body-pair rotation vectors, but reset diagnostics showed thigh/calf angles near
zero even after assembly placed the robot in the home pose.

**Decision:** Compute joint angles from each motor's absolute frames, then apply
the per-joint axis map and sign correction:

```python
frame1 = motor.GetFrame1Abs()
frame2 = motor.GetFrame2Abs()
q_rel = frame1.GetRot().GetInverse() * frame2.GetRot()
```

```python
_JOINT_AXES      = np.array([0,2,2, 0,2,2, 0,2,2, 0,2,2], dtype=np.int32)
_JOINT_AXIS_SIGN = np.where(_JOINT_AXES == 0, 1.0, -1.0)
```

Geometric derivation:

- Hip: URDF X remains Chrono X after the spawn rotation, so sign is +1.
- Thigh/calf: URDF Y maps to Chrono -Z after the -90 degree spawn rotation, so
  the Z rotation vector component must be multiplied by -1.

**Consequences:**

- Pose error is now meaningful at reset.
- Any policy trained before this observation fix is invalid for pose-reward
  tuning.

---

## ADR-009: Observation Space

**Status:** Accepted

**Decision:** 37-dimensional observation:

```text
trunk position       3   (pos.x, pos.y, pos.z)
trunk quaternion     4   (e0, e1, e2, e3)
trunk linear vel     3
trunk angular vel    3
joint angles         12  (FR/FL/RR/RL x hip/thigh/calf)
joint velocities     12
total                37
```

**Consequences:** joint order must stay synchronized with `_JOINT_NAMES`,
`_JOINT_AXES`, `_HOME_JOINT_ANGLES`, and `_JOINT_LOW/HIGH`.

---

## ADR-010: Home-Pose Spawn With `DoAssembly`

**Status:** Accepted

**Context:** `SetRootInitPose()` initializes only the root body. All joint angles
start at zero, placing feet far below the trunk. Motor warm-up loops worked but
wasted training steps.

**Decision:** Set motors to home angles, fix the trunk, run Chrono's kinematic
assembly solver, then unfix the trunk:

```python
self._trunk.SetFixed(True)
system.DoAssembly(1)
self._trunk.SetFixed(False)
```

**Consequences:**

- The robot starts directly in the accepted home pose.
- No 500-step ramp.
- No per-episode warm-up overhead.

---

## ADR-011: Accepted Chrono Home Pose

**Status:** Accepted

**Context:** Position control makes zero action hold the home pose. The original
Menagerie pose (`hip=0.0`, `thigh=0.9`, `calf=-1.8`) sank in this Chrono import.
Reward tuning was compensating for a bad neutral stance.

**Decision:**

```python
home per leg = [0.0, 0.7, -1.4]
spawn height = 0.34
```

**Why:** `less_crouched @ 0.34` starts at its natural support height and remains
stable with zero action. SBEL sign-adjusted candidates also stood, but settled
from a larger drop.

**Consequence:** standing policies should be trained around this baseline.

---

## ADR-012: Foot Support Reward Depends On Contact Forces

**Status:** Accepted

**Context:** Once thigh/calf/hip collisions were disabled, the policy could no
longer hide support in leg links. But it still learned to stand with one foot
unloaded. A foot-height reward would be fragile for SCM because terrain can be
uneven or deformable.

**Decision:** Use low-threshold contact force support:

```python
MIN_FOOT_LOAD = 20.0
foot_contact_penalty = 0.10 * mean(missing_foot_load**2)
```

**Consequences:**

- Encourages all four feet to participate in standing.
- Does not require equal load sharing.
- Does not require equal world foot height.
- Must be revisited on SCM if soil contact force magnitudes change
  significantly.

**Current note:** later reward cleanup raised the active foot-contact penalty to
`2.00`. This ADR explains why contact-force support was introduced; see
`go1_env.py` and `docs/reproduction_ladder.md` for current weights.

---

## ADR-013: Keep Position Motors As The Active Standing Actuator

**Status:** Accepted for the active V3.1 branch

**Context:** The current project phase is focused on learning Project Chrono and
reinforcement-learning workflow around a standing controller. The accepted V3.1
checkpoint was trained and diagnosed with Chrono position motors, and the active
code path is kept aligned with that checkpoint.

**Decision:** Use Chrono position motors as the runtime actuator in `Go1Env`.
The policy outputs normalized home-centered joint-position offsets:

```python
target_q = clip(home_q + 0.20 * executed_action, joint_low, joint_high)
motor_function.SetConstant(target_q)
```

The active home pose is:

```text
[0.0, 0.7, -1.4] per leg
```

**Evidence:** The V3.1 clean-standing diagnostics show the old standing failure
modes, especially foot creep/contact shuffling/action jitter, are controlled in
the clean baseline condition. The active claims stop there: RN1/RN2 reset noise,
random pushes, friction randomization, and observation noise are still future
work.

**Consequences:**

- The simulator and docs now match the V3.1 clean-standing checkpoint.
- Reset-noise and push tests should be interpreted as tests of this
  position-motor standing baseline.
- Torque-actuator work is outside the active branch and is not part of the
  current reproducibility path.

---

## SCM Terrain Parameters

Current parameters from Chrono's deformable-soil demo are starter presets only:

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

These will need tuning against real soil data before SCM training.

## Basic Chrono Physics

Early falling-box smoke tests verified:

```text
ChSystemNSC
Y-up gravity
Bullet collision system
fixed ground box
falling dynamic box
Irrlicht visualization
```

Those tests proved Chrono physics, contact, and visualization worked before the
robot was added.
