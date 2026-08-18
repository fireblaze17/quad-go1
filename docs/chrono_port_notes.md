# Chrono Port Notes

## Coordinate Frame

Chrono is Y-up in this project.

```text
planar body velocity: body X/Z
vertical velocity: body/world Y
yaw rate: about Y
```

The reward and observation code use body-frame planar velocity for command tracking.

This convention is part of the trained baseline. Switching between world-frame and body-frame velocity changes the task the policy is solving.

## Timing

```text
policy_dt: 0.02 s
physics_dt: 0.005 s
substeps: 4
```

The policy target is held for one 50 Hz step. Actuator torques are applied during the 200 Hz physics updates. The actuator-net history is also updated at the 50 Hz policy rate, not at every 200 Hz physics substep.

## Contact

The active environment uses one flat rigid ground. Friction is fixed at `0.8`. The ground is large enough for full default episodes.

Collision diagnostics still report foot, calf, thigh, and trunk contacts for debugging, but the active reward does not include a non-foot contact penalty.

## SCM Terrain

SCM is the active deformable-terrain backend. CRM/SPH was tested, but it is too computationally heavy for practical fine-tuning on this setup.

SCM uses the same Y-up convention as the flat environment:

```text
gravity: (0, -9.81, 0)
SCM reference frame normal: +Y
policy_dt: 0.02 s
physics_dt: 0.005 s
substeps: 4
```

Current SCM soil settings:

```text
solver: BARZILAIBORWEIN
solver iterations: 60
terrain size: 20 x 20
grid spacing: 0.02
Bekker Kphi: 3e6
Bekker Kc: 0
Bekker n: 1.1
cohesion: 0
friction angle: 30 deg
Janosi shear: 0
elastic stiffness: 2e9
damping: 3e4
```

## Actuation

The default actuator is a learned joint actuator model loaded from:

```text
resources/actuator_nets/unitree_go1.pt
```

Input history:

```text
q - q_target at t, t-1, t-2
qdot at t, t-1, t-2
```

History is updated once per policy step and then held during the four physics substeps. Output torque is clipped by URDF effort limits before being sent to Chrono force motors.

The debug actuator is explicit torque-limited joint-space PD:

```text
tau = kp * (q_target - q) - kd * qdot
tau = clip(tau, -effort_limit, effort_limit)
```

## Reward Timing

Reward is computed once per policy step:

```text
reward = 0.02 * raw_reward
```

No positive reward clipping is applied.

See `docs/documentation.md` for the full stack narrative and reproduction commands.
