# Handoff: 48D / 50 Hz Standing Worktree

This is the freshest local working-state brief. If it disagrees with runtime
code or diagnostics JSON, trust `go1_env.py`, `project_config.py`, and the
latest diagnostics artifacts first.

## Active State

```text
environment: 48D relative-state v3.1 plus zero command
actuator: Chrono implicit limited drive
drive gains: Kp=20.0, Kd=0.5 speed setpoint
torque limits: URDF effort limits through ChShaftsClutch
home pose, joint order [FR, FL, RR, RL]:
  FR [-0.1, 0.8, -1.5], FL [0.1, 0.8, -1.5],
  RR [-0.1, 1.0, -1.5], RL [0.1, 1.0, -1.5]
action scale: 0.25
hip action half-scale: indices [0, 3, 6, 9] use effective scale 0.125
action filter: removed from active path
physics timestep: 0.005 s  (200 Hz)
control timestep: 0.020 s  (50 Hz)
physics substeps per policy action: 4
episode length: 1000 RL steps = 20 s simulated time
```

There is no accepted 48D model yet. The active future path in
`project_config.py` is:

```text
runs/stand_v4_implicit_limited_drive_reward_aligned_1m/best_model.zip
```

Historical/source checkpoint only, incompatible with active 48D observation:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
```

That folder says `50k` because it was the planned run folder. The checkpoint was
promoted at 5000 PPO timesteps.

## What Is True Now

Accepted in code design:

```text
no absolute world XYZ in policy input
relative-height termination
implicit limited drive action interface
50 Hz control / 200 Hz physics timing
48D observation
reset-noise sampler support remains available
```

Long-term controller direction:

```text
single flat command-conditioned policy
standing = zero command [vx=0, vz=0, yaw_rate=0]
active observation = 48D, with command inputs currently hardcoded to zero
world model = future auxiliary/prediction component, not an HRL policy switcher
```

Not accepted yet:

```text
clean standing under 48D / 50 Hz / implicit limited drive
RN1/RN2 reset-noise robustness
random push recovery
friction randomization
observation noise
```

Fixed-friction slices are not a current pass condition. Without pushes or other
horizontal disturbances, changing friction does not meaningfully test a standing
policy. Friction randomization should come after push recovery makes friction
matter.

## Reward And Control

The active reward is a zero-command standing adaptation of the closest
`legged_gym` / `walk-these-ways` Go1/A1 reward style. It has no command velocity
sampler and no alive bonus.

```text
tracking_lin_vel_zero   +1.0
tracking_ang_vel_zero   +0.5
lin_vel_y               -2.0
ang_vel_xz              -0.05
orientation             -5.0
base_height             -30.0
torques                 -0.0001
dof_acc                 -2.5e-7
action_rate             -0.01
dof_pos_limits          -10.0
collision               -1.0
dt scaling              0.02 s
positive rewards only   yes
alive reward            no
```

Old custom standing terms are still diagnostics-only with zero reward weight:
foot anchors, loaded-foot slip, base drift, contact switches, foot loads,
raw-action rate, pose/home penalty, and action/control magnitude.

Action interface:

```python
actions_scaled = action * 0.25
actions_scaled[[0, 3, 6, 9]] *= 0.5
target_q = clip(home_q + actions_scaled, joint_low, joint_high)
desired_speed = 20.0 * (target_q - q) - 0.5 * qd
ChShaftsMotorSpeed tracks desired_speed through ChShaftsClutch torque limits
```

Implicit-drive validation:

```text
all 12 driveline motors created
clutch limits match URDF effort limits
positive target error moves every signed joint angle positive
zero action: 835 steps before tip, versus ~40 under raw torque-PD
15.0/0.8 gain sweep is fallback evidence; active path is 20.0/0.5 first
```

## Historical Notes

The 65D v3.1 checkpoint previously fixed clean-standing failure modes under the
old timing: contact shuffling, foot creep, action jitter, unloading, non-foot
support, and base drift. That result is useful evidence, but it must be
re-earned in the active 48D environment.

The 50 Hz control retiming is a modeling correction. Earlier action filtering
and smoothness rewards were partly compensating for a too-fast policy update
rate.

## Canonical Commands

Static check:

```bash
python -m py_compile go1_env.py train_stand.py diagnose_policy.py view_stand_policy.py view_env.py analyze_slip_timeline.py diagnostics.py project_config.py chrono_go1_soil.py
```

Observation-shape check:

```bash
python -c "from go1_env import Go1Env; e=Go1Env(max_steps=1); o,_=e.reset(); print(o.shape, e.observation_space.shape); e.close()"
```

Expected:

```text
(48,) (48,)
```

Train:

```bash
python train_stand.py \
  --save-dir runs/stand_v4_implicit_limited_drive_reward_aligned_1m \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000 \
  --timesteps 1000000 \
  --checkpoint-freq 25000 \
  --learning-rate 0.0003 \
  --clip-range 0.2 \
  --eval-during-training \
  --eval-freq 25000 \
  --eval-episodes 5 \
  --early-stop-patience 5 \
  --early-stop-min-delta 1.0
```

Evaluate:

```bash
python diagnose_policy.py runs/stand_v4_implicit_limited_drive_reward_aligned_1m/checkpoints/stand_policy_25000_steps.zip \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --episodes 30 \
  --max-steps 1000 \
  --reset-noise-level clean \
  --reset-noise-components combined \
  --out diagnostics/v4_implicit_limited_drive_025k_clean30
```

View:

```bash
python view_stand_policy.py runs/stand_v4_implicit_limited_drive_reward_aligned_1m/checkpoints/stand_policy_25000_steps.zip \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000
```

## Next Step

Train and screen clean 48D / 50 Hz / implicit-limited-drive standing first. Do not move to
RN1/RN2 until clean standing passes again.

After that:

```text
1. add nonzero command sampling and train flat command-conditioned locomotion
2. RN1/RN2 reset recovery
3. random push recovery
4. friction randomization after pushes/locomotion make friction meaningful
5. observation noise
6. world model once base behavior is measurable
```
