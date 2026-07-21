# Training Roadmap

This roadmap tracks the active standing-policy path. Historical branches remain
documented in `docs/experiments/`.

The long-term policy architecture is a single flat command-conditioned policy,
not HRL and not separate standing/walking controllers. Standing is the zero
command case. The active observation already includes `[command_vx, command_vz,
command_yaw_rate]`, currently hardcoded to zero for standing.

## Active Environment

```text
observation: 48D relative-state v3.1 plus zero command
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

There is no accepted 48D implicit-limited-drive model yet. The active future checkpoint is:

```text
runs/stand_v4_implicit_limited_drive_reward_aligned_1m/best_model.zip
```

Older position-motor, 65D, and v2 checkpoints are historical/source evidence
only. They are not accepted under the active implicit-limited-drive dynamics.

## Why The Path Changed

Earlier standing work used very fast policy updates. That made action jitter a
major failure mode, and the project responded with stronger smoothness rewards
and an action low-pass filter. The active environment now fixes the interface
directly: physics integrates at 200 Hz, while the policy acts at 50 Hz.

After that correction, the extra 65D observation channels used to support the
old filtered-control setup were removed. The active policy input is 48D: the
45D relative-state observation plus three command inputs. It does not include
previous action, normalized foot loads, or contact flags.

The active reward has also been simplified toward the closest Go1/A1
position-target RL baselines. It uses zero-target velocity tracking, orientation, base height,
torque, action-rate, acceleration, joint-limit, and collision terms. It does not
use an alive bonus or command velocity sampler. The older foot-anchor, foot-slip,
base-drift, and contact-switch terms remain diagnostics only with zero reward
weight.

## Current Work Order

1. Train and evaluate clean 48D / 50 Hz / implicit-limited-drive standing at fixed `mu=0.8`.
2. If clean standing fails, inspect reward terms, contact diagnostics, and
   learning curves before changing reset noise or friction.
3. Add nonzero command sampling and train flat command-conditioned locomotion on flat ground.
4. Test RN1/RN2 reset recovery; isolate failing components before training if needed.
5. Add random push recovery and compare against zero-action behavior.
6. Revisit friction randomization only after pushes or locomotion create
   meaningful horizontal shear demands.
7. Add observation noise after reset and push recovery are understood.
8. Add the world model after base policy behavior is measurable.

RN3 is debug/stretch only and is not an acceptance target.

## Standing Gate

A clean-standing candidate should pass a 30-episode screen before promotion:

```text
survival_rate = 1.0
failure_type_counts nominal-only, or no repeated meaningful failure class
active-reference drift <= 0.03 m
settled total loaded-foot slip <= 0.03 m
settled contact switches = 0 preferred
settled min foot load near/above 20 N
max_nonfoot_load = 0
viewer shows no obvious sliding, chatter, or load collapse
```

Select checkpoints by gate metrics first, not by reward alone.

## Commands

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
