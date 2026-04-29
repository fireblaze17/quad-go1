# Collision Debug Log

Architecture Decision Records for Go1 Chrono collision setup.
Git history records the staged debug process. These docs record only the current rational state.

---

## ADR-001: Collision Whitelist

**Status:** Accepted

**Context:** `ChParserURDF` imports collision shapes with collision disabled.
Enabling all non-fixed bodies caused solver instability (robot launches, scuttles,
then energy explodes) even with motors disabled. Root cause was internal rotor and
sensor marker bodies colliding with terrain and each other before the constraint
solver could converge.

Staged debug using `view_env.py` with `Go1Env(enable_motors=False)`:

```text
feet only                           stable — trunk clips ground
trunk + feet                        stable
trunk + calves + feet               stable
trunk + thighs + calves + feet      stable
trunk + hips + thighs + calves + feet   unstable → fixed by ADR-002
```

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
  Actuator effects are represented through joint parameters (damping, armature, etc.),
  not terrain-contact shells.
- URDF rotor bodies may still contribute mass/inertia but not terrain contact.

---

## ADR-002: Hip Collision Origin

**Status:** Accepted

**Context:** With hips added to the whitelist, the simulation destabilized.
Original hip cylinder:

```xml
<origin rpy="1.5707963267948966 0 0" xyz="0 +/-0.08 0"/>
<cylinder length="0.04" radius="0.046"/>
```

`y=+/−0.08` placed the collision cylinder partially outside the hip body, causing
self-collision and solver instability.

MuJoCo Menagerie reference value for the corresponding hip primitive: `y=+/−0.045`.

**Decision:** Correct the existing element to the Menagerie value:

```xml
<origin rpy="1.5707963267948966 0 0" xyz="0 +/-0.045 0"/>
<cylinder length="0.04" radius="0.046"/>
```

**Consequences:**
- Simulation stable with all five link segments colliding.
- _Rejected: adding extra Menagerie hip primitives_ — tested and destabilised Chrono.
  One cylinder per hip is sufficient.
- Rule: correct values on existing URDF elements only. Do not add new shapes.
