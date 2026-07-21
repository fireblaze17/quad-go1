# Quad Go1

Project Chrono + PPO reinforcement-learning project for a Unitree Go1-style
quadruped controller. The active work is clean standing, but the long-term goal
is a single flat, command-conditioned policy that can stand, walk, and turn. The
goal is not only to train a policy, but to learn the simulation, control,
diagnostics, and RL workflow well enough that each claimed result is
reproducible and inspectable.

This is also a learning project for Project Chrono and reinforcement learning.
Earlier interpretations were revised as the modeling assumptions became clearer;
those corrections are kept in the docs because they explain why the current
workflow exists.

## Current Status

The active code is a new 48D command-conditioned standing environment. There is
no accepted 48D model yet.

```text
active environment:
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

The most important recent correction is the control-rate split: Chrono now
integrates physics at 200 Hz while the policy acts at 50 Hz. Earlier smoothness
rewards and action filtering were partly compensating for updating the policy
too fast. The active setup fixes that on the simulator side first.

The current policy input contains no absolute world XYZ. World position is still
used for diagnostics, reset/reference capture, anchors, and logging, but it is
not fed to the policy.

Old 65D checkpoints are historical/source evidence only and are
shape-incompatible with the active 48D environment:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
```

The directory name says `50k` because that was the planned run folder; the
promoted checkpoint inside it was `5000_steps.zip`.

Not accepted yet under the active 48D setup:

```text
clean standing
RN1/RN2 reset-noise recovery
random push recovery
friction randomization
observation noise
```

## Controller Direction

The project is not targeting an HRL stack or separate standing/walking policies.
The intended controller is one flat policy conditioned on commanded motion:

```text
stand still: command_vx = 0, command_vz = 0, command_yaw_rate = 0
walk/turn:   nonzero command_vx, command_vz, and/or command_yaw_rate
```

The active standing environment already exposes those three command inputs, but
hardcodes them to zero. Once clean standing is reliable, locomotion will replace
the zero command with sampled nonzero commands:

```text
active observation: 48D = previous 45D + [command_vx, command_vz, command_yaw_rate]
```

A world model remains future work for prediction, planning, or representation
learning. It is not intended to be an HRL switcher between separate stand and
walk policies.

## Current Commands

Activate the supported WSL conda environment:

```bash
conda activate chrono-go1
```

Train the current 48D / 50 Hz / implicit-limited-drive setup from scratch:

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

Evaluate a new checkpoint:

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

View a new checkpoint:

```bash
python view_stand_policy.py runs/stand_v4_implicit_limited_drive_reward_aligned_1m/checkpoints/stand_policy_25000_steps.zip \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000
```

## Reward And Control

The active reward now follows the closest Go1/A1 position-target RL baselines more
closely. It is a zero-command standing adaptation of the `legged_gym` /
`walk-these-ways` reward style: no command sampler, no alive bonus, and survival
is implicit because reward only accumulates while the episode is alive.

```text
tracking_lin_vel_zero   +1.0   exp(-(vx^2 + vz^2) / 0.25)
tracking_ang_vel_zero   +0.5   exp(-(yaw_rate^2) / 0.25)
lin_vel_y               -2.0   vertical velocity squared
ang_vel_xz              -0.05  roll/pitch angular velocity squared
orientation             -5.0   projected-gravity tilt error
base_height             -30.0  (relative_height - 0.34)^2
torques                 -0.0001 sum(motor_torque^2)
dof_acc                 -2.5e-7 sum(((joint_vel - prev_joint_vel) / dt)^2)
action_rate             -0.01  sum((action - previous_action)^2)
dof_pos_limits          -10.0  joint-limit violation amount
collision               -1.0   non-foot contact indicator
dt scaling              reward scales are multiplied by 0.02 s
positive rewards only   final reward is clipped at zero
```

Old custom standing terms such as foot anchors, loaded-foot slip, base drift,
contact switches, foot loads, and raw-action diagnostics are still logged for
debugging, but their reward weights are `0.0` in the active reward.

The policy outputs normalized joint-position offsets, but the runtime actuator
is now a Chrono driveline-based implicit limited drive:

```python
actions_scaled = action * 0.25
actions_scaled[[0, 3, 6, 9]] *= 0.5
target_q = clip(home_q + actions_scaled, joint_low, joint_high)
desired_speed = 20.0 * (target_q - q) - 0.5 * qd
ChShaftsMotorSpeed tracks desired_speed through ChShaftsClutch torque limits
```

Without the hip half-scale, early torque-PD trials could drive the legs too far
laterally and produce a splits-like stance. After checking common Go1-style RL
action scaling patterns, the hip action range was reduced by 50% while keeping
thigh/calf authority higher.

The raw explicit torque-PD branch was unstable/weak for zero action. The
implicit-limited-drive validation created all 12 driveline motors, matched
clutch limits to URDF effort limits, and moved every joint in the correct
direction for a positive target error. The active gains are the common
Go1/A1-style baseline `Kp=20.0`, `Kd=0.5`; a `15.0/0.8` sweep is documented as
a fallback candidate, but the active path returns to `20.0/0.5` first. This is
actuator validation only, not an accepted trained standing policy.

## Main Issues Encountered

The main lesson so far is that survival alone is weak evidence. Standing must be
judged by drift, loaded-foot slip, contact switching, foot load, friction usage,
non-foot contacts, action behavior, and observation design.

Key experiment decisions:

- Long-hold foot creep after apparently stable standing:
  [ADR-002](docs/experiments/fixed_friction_standing.md#adr-002-anchor5-improved-support-but-failed-long-hold-creep)
- Action jitter causing slip under the old fast-control setup:
  [ADR-005](docs/experiments/fixed_friction_standing.md#adr-005-freeze-action-diagnostic-identified-action-jitter)
- Rejected normalized foot-slip and stance-shape branches:
  [ADR-007](docs/experiments/fixed_friction_standing.md#adr-007-normalized-foot-slip-005-did-not-help),
  [ADR-008](docs/experiments/fixed_friction_standing.md#adr-008-stance-shape-005-and-0005-were-rejected)
- Absolute world XYZ removed from the policy input:
  [ADR-015](docs/experiments/fixed_friction_standing.md#adr-015-relative-state-standing-v3-attempt-stopped-at-fixed-mu-gate)
- V3.1 65D clean-standing recovery, now historical/source evidence:
  [ADR-016](docs/experiments/fixed_friction_standing.md#adr-016-v31-filter-state-and-slip-aligned-reward)
- Friction-slice interpretation corrected:
  [ADR-018](docs/experiments/fixed_friction_standing.md#adr-018-friction-slice-claim-retracted-until-pushes-make-friction-meaningful)
- Physics/control-rate correction:
  [ADR-020](docs/experiments/fixed_friction_standing.md#adr-020-separate-physics-rate-from-policy-control-rate)
- 65D to 48D observation reduction:
  [ADR-021](docs/experiments/fixed_friction_standing.md#adr-021-reduce-active-observation-from-65d-back-to-45d)
- Raw torque-PD actuator and hip half-scale, followed by the active Chrono
  implicit-limited-drive actuator:
  [ADR-022](docs/experiments/fixed_friction_standing.md#adr-022-raw-torque-limited-pd-branch-was-replaced)
  and
  [ADR-024](docs/experiments/fixed_friction_standing.md#adr-024-test-chrono-driveline-based-implicit-limited-drive)

## Next Work

Current progression:

```text
1. train and evaluate clean 48D / 50 Hz standing
2. add nonzero command sampling once standing is stable
3. train flat command-conditioned locomotion on flat ground
4. define, test, and possibly train RN1/RN2 reset recovery
5. add random push recovery
6. revisit friction randomization after pushes/locomotion make friction meaningful
7. add observation noise
8. add a world model after base policy behavior is measurable
```

RN3 is temporary/debug-only and is not a formal accepted reset-noise level.

## Docs Map

- [docs/reproducibility.md](docs/reproducibility.md) - active copy-paste commands
- [docs/reproduction_ladder.md](docs/reproduction_ladder.md) - reproduction order and lineage
- [docs/training_roadmap.md](docs/training_roadmap.md) - current research direction
- [docs/experiments/fixed_friction_standing.md](docs/experiments/fixed_friction_standing.md) - standing ADR log
