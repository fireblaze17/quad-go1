# Collision Debug Log

This log records the collision decisions for the Chrono Go1 port.

## Problem

The initial contact fix enabled collision on every non-fixed body imported by
`ChParserURDF`. That proved the robot could contact terrain, but it made the
simulation unstable:

```text
robot launches upward
or scuttles briefly, then solver energy explodes
```

This happened even when motors were disabled, so the cause was not only
position target snapping.

## Debug Method

`view_env.py` was used as the live test harness.

`Go1Env` also supports:

```python
Go1Env(enable_motors=False)
```

This isolates URDF/contact behavior from motor targets.

Collision was re-enabled in stages:

```text
feet only                  stable, but trunk clips through ground
trunk + feet               stable
trunk + calves + feet      stable
trunk + thighs + calves + feet
                           stable
trunk + hips + thighs + calves + feet
                           unstable until hip origin fix
```

## Current Collision Whitelist

The env disables all robot body collision first, then enables only the external
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

Disabled collision bodies include:

```text
hip rotors
thigh rotors
calf rotors
camera marker links
ultrasound marker links
other tiny sensor/ROS marker collision boxes
```

## Why Rotor/Sensor Collisions Stay Disabled

The project runs in Chrono, but MuJoCo Menagerie is used as a reference model
for Go1 values. In Menagerie `unitree_go1/go1.xml`, collision geoms are attached
to external bodies:

```text
trunk
hip
thigh
calf
foot
```

There are no separate colliding rotor bodies such as:

```text
FR_hip_rotor
FR_thigh_rotor
FR_calf_rotor
```

Menagerie represents rotor/actuator effects through joint and actuator
parameters such as damping, friction loss, armature, position gains, and force
ranges. It does not expose rotor cylinders as terrain-contact shells.

Tradeoff: the URDF rotor bodies may still contribute imported mass/inertia, but
their collision shapes are excluded from terrain and self-contact. This matches
the external-contact intent better than letting internal motor markers collide.

References:

```text
MuJoCo Menagerie Go1:
https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_go1/go1.xml

MuJoCo body/geom docs:
https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom

MuJoCo joint docs, including armature:
https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint
```

## Hip Collision Fix

The original Chrono URDF had one hip collision cylinder per hip body:

```xml
<origin rpy="1.5707963267948966 0 0" xyz="0 +/-0.08 0"/>
<cylinder length="0.04" radius="0.046"/>
```

With hips enabled, this destabilized the simulation.

MuJoCo Menagerie's closest corresponding hip primitive uses:

```text
position: +/-0.045
radius:   0.046
```

The Chrono URDF fix changes the existing collision element in place:

```xml
<origin rpy="1.5707963267948966 0 0" xyz="0 +/-0.045 0"/>
<cylinder length="0.04" radius="0.046"/>
```

Important constraint:

```text
Do not add new collision elements just because Menagerie has extra primitives.
Only correct values on existing URDF collision elements unless there is a clear
Chrono-specific reason to modify the model structure.
```

Extra Menagerie hip cylinders were briefly tested and destabilized Chrono, so
they were removed. The current URDF keeps one hip collision cylinder per hip.

## Current Result

With:

```text
position motors enabled
target ramp enabled
trunk/hip/thigh/calf/foot collisions enabled
rotor/sensor collisions disabled
hip origin corrected to +/-0.045
```

`view_env.py` shows the robot standing without the previous launch/explosion
behavior.
