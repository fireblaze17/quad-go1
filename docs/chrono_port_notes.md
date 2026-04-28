# Chrono Port Notes

This document keeps the detailed migration notes for the Project Chrono Go1
port. The README is the front-door narrative; this file is the longer engineering
log.

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

## URDF Import

The Go1 URDF and mesh assets live under:

```text
models/go1/
```

The original URDF referenced ROS package paths such as:

```text
package://go1_description/meshes/trunk.dae
```

These were converted to local paths:

```text
meshes/trunk.dae
```

## Chrono-Specific URDF

The source URDF included a dummy `base` link and a fixed `floating_base` joint.
Chrono imported the root as fixed, making the robot hang rather than fall.

The Chrono-specific URDF removes that dummy fixed root path:

```text
models/go1/go1_chrono.urdf
```

Current intended root:

```text
root body: trunk
root fixed: False
```

## Z-Up To Y-Up

The Go1 URDF follows ROS convention and is Z-up. Chrono is used as Y-up in this
project. The import applies:

```python
chrono.QuatFromAngleX(-math.pi / 2)
```

This maps URDF Z-up standing orientation into the Chrono Y-up world.

## Collision Activation

`ChParserURDF` builds collision shapes from URDF `<collision>` tags but leaves
collision disabled on imported bodies. Collision must be explicitly enabled
after `PopulateSystem()`.

The first broad fix enabled collision on all non-fixed bodies. Later debugging
refined this into the collision whitelist described in
[collision_debug_log.md](collision_debug_log.md).

## Parser Configuration

The env currently uses:

```python
parser.SetAllBodiesMeshCollisionType(
    parsers.ChParserURDF.MeshCollisionType_TRIANGLE_MESH
)

parser.SetAllJointsActuationType(
    parsers.ChParserURDF.ActuationType_POSITION
)
```

Triangle mesh collision was chosen over convex hull to preserve the local shape
of curved Go1 link geometry. Position actuation was chosen for the first
standing and walking baseline.

## Contact Materials

Material values are taken from MuJoCo Menagerie as reference values:

```text
default body friction: 0.6
foot friction:         0.8
restitution:           0.0
```

`ChContactMaterialData` is used with the URDF parser. Foot material overrides
must be set before `PopulateSystem()`.

## Gymnasium Env

`go1_env.py` wraps the Chrono simulation as a Gymnasium env.

Observation:

```text
trunk position       3
trunk quaternion     4
trunk linear vel     3
trunk angular vel    3
joint angles         12
joint velocities     12
total                37
```

Action:

```text
12 normalized joint-position offsets in [-1, 1]
target = home_joint_angles + 0.25 * action
```

The joint order is:

```text
FR hip/thigh/calf
FL hip/thigh/calf
RR hip/thigh/calf
RL hip/thigh/calf
```

## Reset And Visualization

SCM terrain deformation cannot be cleared in place, so `reset()` rebuilds the
entire Chrono system.

`ChVisualSystemIrrlicht` cannot be safely reattached to a rebuilt system after
initialization, so the env creates a fresh visualizer each build when
`render_mode="human"`.

Tradeoff: full rebuilds are slower, but the reset logic is reliable for both
flat and SCM terrain.
