# Reproduction Ladder

This guide records the closest reproduction order from an untrained standing
policy to the current active 48D worktree. Run artifacts under `runs/` are
gitignored, so exact checkpoint reproduction requires sharing those artifacts
out of band.

## Current Active Target

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
active future model: runs/stand_v4_implicit_limited_drive_reward_aligned_1m/best_model.zip
```

There is no accepted 48D implicit-limited-drive model yet.

The intended final controller is one flat command-conditioned policy. Standing
is the zero-command case. The active 48D observation already includes three
command inputs (`vx`, `vz`, yaw rate), currently hardcoded to zero, instead of
introducing an HRL stand/walk switch.

## Stage 0: Environment Check

```bash
conda activate chrono-go1
python -m py_compile go1_env.py train_stand.py diagnose_policy.py view_stand_policy.py view_env.py analyze_slip_timeline.py diagnostics.py project_config.py chrono_go1_soil.py
python -c "from go1_env import Go1Env; e=Go1Env(max_steps=1); o,_=e.reset(); print(o.shape, e.observation_space.shape); e.close()"
```

Expected observation shape:

```text
(48,) (48,)
```

## Stage 1: Historical V2 Standing

Historical purpose: first working fixed-friction standing controller.

Approximate original command:

```bash
python train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000 --seed 1 --save-dir runs/stand
```

What it taught:

```text
survival is not enough
settled drift, loaded-foot slip, contact switches, and non-foot contacts matter
```

Status: historical. Do not use v2 as the active baseline.

## Stage 2: Historical Support/Jitter Cleanup

Historical purpose: diagnose why visually stable policies still crept over long
holds.

Key outcomes:

```text
anchor/support diagnostics exposed long-hold creep
freeze-action diagnostic identified action jitter
jitter-suppression fine-tune improved clean standing
normalized foot-slip and stance-shape reward branches were rejected
action filtering worked under the old fast-control setup
```

Status: historical evidence. The active code no longer uses the action filter.

## Stage 3: Historical V3/V3.1 Relative-State Work

Historical purpose: remove absolute world XYZ from the policy input.

```text
v3 35D: removed absolute XYZ but failed slip gate
v3.1 65D: added filter-state/load/contact inputs and slip-aligned rewards
```

Historical source checkpoint:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
```

This checkpoint was clean-standing evidence under the old timing, but it is
shape-incompatible with the active 48D environment.

## Stage 4: Friction Interpretation Correction

Fixed-friction slices without pushes were reclassified as not meaningful for
standing robustness. A clean standing pose does not necessarily demand
horizontal shear, so changing friction alone may not test the policy. Friction
randomization should come after random push recovery exists.

Status: corrected interpretation, not an accepted robustness claim.

## Stage 5: Sim/Control-Rate Correction

Active correction:

```text
physics integration: 0.005 s / 200 Hz
policy control: 0.020 s / 50 Hz
substeps per policy action: 4
1000 RL steps: 20 s simulated time
```

Reason: smoothness should be handled first by a sane simulator/control-rate
split, not only by reward shaping and action filtering.

Status: implemented. Requires new training, and old position-motor checkpoints
are easier-actuator history.

## Stage 6: 65D To 45D Observation Reduction

Active observation:

```text
1   base height relative to ground/support
4   trunk quaternion
3   base linear velocity
3   base angular velocity
12  joint positions
12  joint velocities
2   support-frame base X/Z error from active standing reference
8   support-frame per-foot anchor X/Z errors
---
45 total
```

Removed from the old 65D observation:

```text
12 previous executed action
4 normalized foot loads
4 contact flags
```

Status: implemented. Existing 65D checkpoints are incompatible, and older
position-motor checkpoints are not accepted under the implicit-limited-drive dynamics.

## Stage 6.1: Add Command Inputs For Flat Policy

Active observation is now 48D:

```text
45D relative-state standing observation
3 command inputs: command_vx, command_vz, command_yaw_rate
---
48 total
```

Status: implemented with zero commands for standing. Requires new training.

## Stage 6.5: Align Reward With Go1/A1 Implicit-Limited-Drive Baselines

Active reward:

```text
tracking_lin_vel_zero, tracking_ang_vel_zero, lin_vel_y, ang_vel_xz,
orientation, base_height, torques, dof_acc, action_rate, dof_pos_limits,
collision
```

There is no alive reward and no command velocity sampler. The zero velocity
target is hardcoded for standing. Older custom terms such as foot anchors,
loaded-foot slip, base drift, contact switches, and foot-load penalties remain
diagnostics only with zero reward weight.

Status: implemented. Requires new training.

## Stage 7: Train Active 45D Model

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

Promote by diagnostics, not training reward alone.

Clean screen:

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

Pass gate:

```text
survival_rate = 1.0
active-reference drift <= 0.03 m
settled total loaded-foot slip <= 0.03 m
settled contact switches = 0 preferred
settled min foot load near/above 20 N
max_nonfoot_load = 0
no repeated meaningful failure class
```

## Stage 8: Next After Clean 45D Standing

Do these only after Stage 7 passes:

```text
1. add nonzero command sampling
2. train flat command-conditioned locomotion on flat ground
3. RN1/RN2 reset recovery
4. random push recovery
5. friction randomization after pushes/locomotion make friction meaningful
6. observation noise
7. world model once base behavior is measurable
```

RN3 is debug/stretch only.
