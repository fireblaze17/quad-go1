# Collision Debug Log

Architecture Decision Records for Go1 Chrono collision setup.

Git history records the staged debug process. This file records the current
reasoning, including the later policy-contact diagnostics that changed the
collision whitelist.

---

## ADR-001: Initial Collision Bring-Up

**Status:** Historical

**Context:** `ChParserURDF` imports collision shapes with collision disabled.
Enabling all non-fixed bodies caused solver instability: the robot launched,
scuttled, and eventually exploded even with motors disabled.

Staged debug using `view_env.py` with `Go1Env(enable_motors=False)`:

```text
feet only                                stable, trunk clips ground
trunk + feet                             stable
trunk + calves + feet                    stable
trunk + thighs + calves + feet           stable
trunk + hips + thighs + calves + feet    initially unstable
```

The first conclusion was that a full external envelope could be made stable if
internal rotor, camera, and sensor marker bodies stayed disabled.

**Why this later changed:** zero-action stability was not the same as policy
training quality. The full external envelope let the policy exploit non-foot
leg contacts.

---

## ADR-002: Hip Collision Origin

**Status:** Historical fix, kept in URDF

**Context:** With hips added to the old whitelist, the simulation destabilized.
Original hip cylinder:

```xml
<origin rpy="1.5707963267948966 0 0" xyz="0 +/-0.08 0"/>
<cylinder length="0.04" radius="0.046"/>
```

`y=+/-0.08` placed the collision cylinder too far outboard.

**Decision:** Correct the existing element to the Menagerie-like value:

```xml
<origin rpy="1.5707963267948966 0 0" xyz="0 +/-0.045 0"/>
<cylinder length="0.04" radius="0.046"/>
```

**Consequences:**

- This made the old full external envelope more stable.
- Hip collision is no longer enabled in the training whitelist, but the geometry
  correction is still better if hips are ever re-enabled.

---

## ADR-003: Policy Viewer Foot Slip Diagnostics

**Status:** Accepted

**Context:** The trained policy stood but visibly shuffled. It was unclear
whether this was reward chatter, motor dynamics, or actual foot slip.

**Decision:** Add always-on viewer diagnostics:

```text
foot_dxz_mean/max    displacement from reset X/Z position
foot_vxz_mean/max    horizontal foot speed
foot_y               world Y foot height
foot_share           normalized foot contact-force share
foot_load            absolute Y contact force per foot
foot_y_min/max       interval foot height range
foot_load_min/max    interval foot load range
```

**What it showed:** foot displacement and velocity were real. This was not just
a camera or mesh illusion.

**Tradeoff:** the viewer output is verbose, but it is intentionally a diagnostic
tool. Evaluation remains the compact summary.

---

## ADR-004: Leg-Link Contact Diagnostics

**Status:** Accepted

**Context:** The front-left leg looked raised, but foot load was sometimes still
nonzero. We needed to know whether other leg links were carrying hidden support.

**Decision:** Add viewer-only contact-force diagnostics for every leg group:

```text
foot_load
calf_load
thigh_load
hip_load
leg_nonfoot_load
nonfoot_load_max
```

These read `abs(GetContactForce().y)` because the world is Y-up.

**Result with old full leg collisions:**

```text
calf_load/thigh_load: thousands of newtons
leg_nonfoot_load:     about 12000-16000 N per leg
foot_load:            about 150-225 N
```

**Interpretation:** the policy was partly supported by calf/thigh collision
geometry. That explained why reward tuning alone could not cleanly fix the
stance.

---

## ADR-005: Final Training Collision Whitelist

**Status:** Accepted

**Decision:** Disable all robot collision after import, then enable only:

```python
_ROBOT_COLLISION_BODIES = (
    "trunk",
    "FR_foot", "FL_foot", "RR_foot", "RL_foot",
)
```

Hips, thighs, calves, rotors, camera bodies, and sensor marker bodies stay
non-colliding.

**Why this worked:** normal standing and walking should be supported by feet.
Removing hip/thigh/calf terrain collisions removes a hidden support exploit.
This follows the relevant MaGIC 2025 Chrono lesson: disable non-foot leg
collisions for robot locomotion training unless those contacts are explicitly
the subject of the study.

**Tradeoffs:**

- Less literal during falls, scrapes, and leg-terrain impacts.
- Better for learning foot-ground support.
- Trunk remains collidable, so fallen-body contact still exists.
- Diagnostics remain in `view_stand_policy.py` so accidental non-foot contact
  can be caught immediately.

---

## ADR-006: Foot-Only Contact Exposed A New Failure

**Status:** Solved by reward, not collision

**Context:** After disabling non-foot leg collisions, the old policy still
survived but one foot, usually front-left, could remain high and unloaded.
Viewer output showed:

```text
FL foot_load: 0.0
FL foot_y:    higher than the other feet
nonfoot_load: 0.0
```

**Interpretation:** this was no longer hidden leg collision. It was a support
objective problem: survival, uprightness, tilt, pose, symmetry, and smoothness
still did not require every foot to participate.

**Decision:** Add a low-threshold four-foot support penalty in the reward:

```text
minimum foot load:    20 N
penalty weight:       0.10
```

**Why it worked:** it directly rewards the missing physical requirement. The
accepted policy now has all four feet near the ground, all feet carrying load,
and zero non-foot contact.

---

## Final Accepted Contact Signature

In the accepted standing baseline, the viewer should look like:

```text
calf_load=(FR:+0.0,FL:+0.0,RR:+0.0,RL:+0.0)
thigh_load=(FR:+0.0,FL:+0.0,RR:+0.0,RL:+0.0)
hip_load=(FR:+0.0,FL:+0.0,RR:+0.0,RL:+0.0)
nonfoot_load_max=(FR:+0.0,FL:+0.0,RR:+0.0,RL:+0.0)
foot_y all close after settling
no foot permanently at zero load
```

Example accepted evaluation indicators:

```text
survival_rate:      1.000
mean_length:        1000.0
min_upright_score:  1.000
mean_foot_load:     32.09 N
min_foot_load:      17.58 N
foot_contact_error: 0.020
```

## Later Standing-Quality Note

Later drift/slip failures are not the old non-foot collision exploit returning.
Diagnostics continue to report zero calf/thigh/hip support. The current
standing-quality issues are policy problems: horizontal creeping, contact
quality, load bias, and occasional persistent lean while supported only by trunk
+ feet collision.
