# Quad Go1

Project Chrono + PPO reinforcement-learning project for a Unitree Go1-style
quadruped. The project is building a reproducible standing controller before
moving to terrain and locomotion.

The active worktree is now standing v3.1: a 65D relative-state observation with
no absolute world XYZ in the policy input and relative-height termination. The
first v3.1 fixed-friction checkpoint is accepted for clean flat standing and
coordinate-invariance checks.

## Current V3.1 Result

```text
current v3.1 fixed-standing baseline:
  runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip

required runtime setup:
  action_filter_tau = 0.05
  foot_friction = 2.0

accepted so far:
  effective flat mu = 0.5-1.2
  clean reset
  spawn X/Z offsets through +/-0.5 m
  ground-height offsets through +/-0.20 m
```

V3.1 has not yet been re-accepted for reset noise. The last accepted v2
checkpoint still owns the RN-1/RN-2 reset-noise claim.

V3.1 fixed `mu=0.8`, clean reset confirmation:

```text
episodes: 30
result: 30/30 nominal
active-reference drift: 0.000207 m
settled total contact foot slip: 0.007678 m
settled contact switches: 0
settled min foot load: 26.95 N
max settled friction usage: 0.03711
max non-foot load: 0.0
```

Coordinate-invariance screening also passed:

```text
spawn offsets: 8 shifted X/Z cases, 30/30 nominal each
ground-height offsets: -0.20, -0.10, +0.10, +0.20 m, 30/30 nominal each
worst coordinate-screen drift: 0.000288 m
worst coordinate-screen slip: 0.008764 m
switches: 0 in every screen
```

V3.1 friction keeper confirmation passed without extra PPO training. Each row
below is a fixed effective-friction slice using the same checkpoint, clean reset,
flat terrain, `foot_friction=2.0`, and `action_filter_tau=0.05`:

```text
mu   episodes  nominal  drift_m   slip_m    switches  min_load_N  max_friction_usage  worst_foot
0.5  100       100/100   0.000201  0.007565  0         26.946      0.034239            RR
0.6  100       100/100   0.000200  0.007615  0         26.951      0.026631            RR
0.8  100       100/100   0.000215  0.007392  0         26.977      0.025404            RR
0.9  100       100/100   0.000215  0.007392  0         26.977      0.022581            RR
1.0  100       100/100   0.000215  0.007392  0         26.977      0.020323            RR
1.1  100       100/100   0.000215  0.007392  0         26.977      0.018476            RR
1.2  100       100/100   0.000215  0.007392  0         26.977      0.016936            RR
```

## Last Accepted V2 Result

```text
last accepted v2 baseline:
  runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip

required runtime setup:
  action_filter_tau = 0.05
  foot_friction = 2.0

accepted effective friction range:
  mu = 0.5-1.2

accepted reset range:
  clean, RN-1, RN-2
```

The v2 checkpoint alone is not the full controller. Reproducing the accepted v2
behavior requires the environment action filter and the `2.0` foot-friction
cap-removal setting. It also requires the pre-v3 37D observation code; v2
checkpoints are shape-incompatible with the active v3.1 worktree.

Thirty deterministic 5000-step episodes were run at each fixed slice:

```text
slices: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
result: 30/30 nominal on every slice
worst active-reference drift: 0.001558 m
worst settled total foot slip: 0.003871 m
settled contact switches: 0 on every slice
worst settled min foot load: 28.53 N
max settled friction usage: 0.01833
```

Reset-noise keeper confirmation used 100 deterministic 5000-step episodes for
each reset/friction pair:

```text
reset levels: clean, RN-1, RN-2
friction slices: 0.5, 0.8, 1.2
result: 100/100 nominal on every condition
worst active-reference drift: 0.002813 m
worst settled total foot slip: 0.003756 m
settled contact switches: 0 on every condition
worst settled min foot load: 28.53 N
max settled friction usage: 0.02035
```

Strongest single RN-2 evidence file:

```text
diagnostics/keeper_reset_rn2_mu_0p5_confirm100/summary.json
```

## Quick Start

Activate the supported WSL conda environment:

```bash
conda activate chrono-go1
```

View the accepted v3.1 fixed-standing checkpoint:

```bash
python view_stand_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 5000 --action-filter-tau 0.05
```

Run the accepted fixed-standing diagnostic:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --out diagnostics/v3p1_fixed08_005k_clean_mu08_confirm30
```

Run a coordinate-invariance check:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --spawn-x 0.5 --spawn-z 0.5 --out diagnostics/v3p1_fixed08_005k_spawn_x0p5_z0p5_confirm30
```

Reproduce the low-friction edge of the accepted V3.1 range:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.5 --friction-max 0.5 --episodes 100 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --out diagnostics/v3p1_friction_keeper_mu_0p5_confirm100
```

## Reward And Control

The active v3.1 reward rewards survival/uprightness and penalizes tilt, pose
error, control effort, joint velocity, action rate, angular velocity, raw X/Z
base velocity, missing foot load, direct loaded-foot slip, post-settle
foot-anchor drift, post-settle base drift, contact switches, anchor resets,
anchor deactivations, raw action jitter, and filter lag.

Important active settings:

```text
alive_bonus             1.00
upright                 0.15
pose                    0.30
control                 0.03
joint_velocity          0.02
action_rate             0.05
raw_action_rate         0.02
filter_lag              0.02
tilt                    0.25
angular_velocity        0.01
xz_velocity             1.00
foot_contact            mean 1.00, worst-foot 2.00
foot_slip               50.00 * loaded_step_slip / 0.03 m
foot_anchor             0.10 normalized beyond 0.005 m
base_drift              0.05 normalized beyond 0.01 m
contact_switch          0.10 per hysteresis switch
anchor_reset            0.50 per reset
anchor_deactivation     1.00 per deactivation
load quality ramp       first 50 steps
stance quality ramp     first 100 steps
minimum foot load       20 N
```

The action filter is part of the control interface, not a reward term:

```python
alpha = dt / (tau + dt)
executed_action = previous_executed_action + alpha * (raw_action - previous_executed_action)
```

For the last accepted v2 baseline, `tau=0.05` and `dt=0.002`, so
`alpha=0.038462`.

Rejected reward paths are documented in the ADR log. Stance-shape, normalized
X/Z velocity, base drift `10.0`, and global foot-slip reward `0.05` are not
active.

## Main Issues Encountered

The main project lesson is that “survives the episode” was not enough. The
standing policy had to be judged by settled drift, loaded-foot slip, contact
switching, foot load, friction usage, and whether the policy depended on world
coordinates.

Key issues and where they are documented:

- Long-hold foot creep after apparently stable standing:
  [ADR-002](docs/experiments/fixed_friction_standing.md#adr-002-anchor5-improved-support-but-failed-long-hold-creep)
- Over-strong base drift reward `10.0` worsening contact behavior:
  [ADR-003](docs/experiments/fixed_friction_standing.md#adr-003-base-drift-100-was-too-strong)
- Action jitter causing slip even when the standing pose was viable:
  [ADR-005](docs/experiments/fixed_friction_standing.md#adr-005-freeze-action-diagnostic-identified-action-jitter)
- Rejected normalized foot-slip and stance-shape reward branches:
  [ADR-007](docs/experiments/fixed_friction_standing.md#adr-007-normalized-foot-slip-005-did-not-help),
  [ADR-008](docs/experiments/fixed_friction_standing.md#adr-008-stance-shape-005-and-0005-were-rejected)
- Absolute world XYZ in the old v2 observation:
  [ADR-015](docs/experiments/fixed_friction_standing.md#adr-015-relative-state-standing-v3-attempt-stopped-at-fixed-mu-gate)
- V3.1 recovery with relative observation, filter-state input, and slip-aligned
  reward: [ADR-016](docs/experiments/fixed_friction_standing.md#adr-016-v31-filter-state-and-slip-aligned-reward)
- V3.1 friction `0.5-1.2` accepted without extra training:
  [ADR-017](docs/experiments/fixed_friction_standing.md#adr-017-v31-friction-robustness-passed-without-ppo-training)

## Docs Map

- [docs/reproducibility.md](docs/reproducibility.md) - copy-paste commands
- [docs/reproduction_ladder.md](docs/reproduction_ladder.md) - closest path from untrained policy to current result
- [docs/training_roadmap.md](docs/training_roadmap.md) - current research direction
- [docs/experiments/fixed_friction_standing.md](docs/experiments/fixed_friction_standing.md) - ADR log for accepted/rejected standing experiments
- [docs/chrono_port_notes.md](docs/chrono_port_notes.md) - Chrono import, contact, physics, and material decisions
- [docs/collision_debug_log.md](docs/collision_debug_log.md) - collision/contact debugging history
- [docs/experiments/friction_curriculum.md](docs/experiments/friction_curriculum.md) - archived pre-filter friction history
- [docs/experiments/standing_v2.md](docs/experiments/standing_v2.md) - historical fixed-friction model card
- [HANDOFF.md](HANDOFF.md) - local working-state brief

## Project Shape

```text
go1_env.py                 Chrono Gymnasium environment and reward
train_stand.py             PPO standing-policy training
evaluate_stand.py          headless policy evaluation
diagnose_policy.py         settled-window and timeline diagnostics
analyze_slip_timeline.py   foot-slip timeline classifier
run_regression.py          evaluation + diagnosis wrapper
compare_friction_slices.py fixed-friction slice comparison
view_stand_policy.py       trained-policy Irrlicht viewer
view_env.py                zero-action/live environment viewer
friction_curriculum.py     helper command printer for friction checks/history
project_config.py          shared paths and defaults
chrono_go1_soil.py         SCM deformable terrain milestone
docs/                      decision logs and reproducibility notes
```

## Next Work

Friction robustness is now accepted for v3.1. RN-1/RN-2 reset-state robustness
is still accepted only for the v2 baseline. The next step is to re-run the
reset-noise gates on v3.1 before observation noise.

Next sequence:

```text
1. re-check reset-noise gates for v3.1
2. add observation noise
3. move to SCM/deformable terrain bridge
4. build locomotion policies
```

Every new checkpoint should be rechecked on fixed friction slices before it is
accepted. Promote by worst-slice behavior, not average reward alone.

## Nominal Load Reference

The zero-action home-pose load reference at fixed `mu=0.8` is:

```text
FR: 27.17%
FL: 27.55%
RR: 22.46%
RL: 22.82%
front/rear: 54.7/45.3
left/right: 50.4/49.6
```

Exact `25/25/25/25` loading is not the natural target for the current home pose.
Left/right balance remains a useful diagnostic.
