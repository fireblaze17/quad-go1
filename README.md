# Quad Go1

Project Chrono robotics simulation and reinforcement-learning project for a
Unitree Go1-style quadruped.

## Goal

1. Build a reproducible Go1 simulation in Chrono.
2. Wrap it as a Gymnasium environment.
3. Train clean standing on flat terrain.
4. Expand the accepted standing controller to randomized friction.
5. Transfer to Chrono SCM deformable terrain and later hierarchical skills.

## Current Status

```text
Stage: fixed-friction standing accepted; friction bridge checks next

Current baseline checkpoint:
  runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip

Required control setup:
  action_filter_tau = 0.05

Fixed mu=0.8 confirmation:
  episodes: 30/30 nominal
  active-reference drift: 0.001637 m
  settled total foot slip: 0.003516 m
  settled contact switches: 0
  settled min foot load: 26.68 N
```

The accepted baseline is the checkpoint plus the environment action filter.
The checkpoint alone is not the full control recipe.

## Docs Map

- [docs/reproducibility.md](docs/reproducibility.md) - copy-paste commands
- [docs/training_roadmap.md](docs/training_roadmap.md) - current decisions and next work
- [docs/experiments/fixed_friction_standing.md](docs/experiments/fixed_friction_standing.md) - fixed-standing ADR log
- [docs/experiments/friction_curriculum.md](docs/experiments/friction_curriculum.md) - archived pre-filter friction history
- [docs/experiments/standing_v2.md](docs/experiments/standing_v2.md) - historical fixed-friction model card
- [docs/collision_debug_log.md](docs/collision_debug_log.md) - collision/contact ADRs
- [docs/chrono_port_notes.md](docs/chrono_port_notes.md) - Chrono import, solver, and contact ADRs
- [HANDOFF.md](HANDOFF.md) - freshest working-state brief

## Active Reward And Control

The active standing reward in `go1_env.py` is:

```text
reward =
  alive_bonus
+ upright_reward
- tilt_penalty
- pose_penalty
- control_penalty
- joint_velocity_penalty
- action_rate_penalty
- angular_velocity_penalty
- raw X/Z velocity penalty
- missing-foot-load contact penalty
- planted-foot anchor penalty after step 100
- base drift penalty after step 100
```

Current important settings:

```text
alive_bonus:             1.00
upright:                 0.15
pose:                    0.30
control:                 0.03
angular_velocity:        0.01
xz_velocity:             1.00
joint_velocity:          0.02
action_rate:             0.05
tilt:                    0.25
foot_contact:            2.00
foot_slip:               0.00
foot_anchor:             5.00, 0.005 m deadband
base_drift:              2.00, 0.01 m deadband
action_filter_tau:       0.05 when running accepted baseline
```

The action filter is a control-interface filter, not a reward term. The policy
outputs raw actions; the environment low-pass filters them before setting motor
targets. Stance-shape, normalized X/Z velocity, base drift `10.0`, and global
foot-slip reward `0.05` are rejected experiments, not active baseline behavior.

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
friction_curriculum.py     helper commands for bridge/friction work
project_config.py          shared paths and defaults
chrono_go1_soil.py         SCM deformable terrain milestone
mujoco/                    historical MuJoCo reference
docs/                      decision logs and reproducibility notes
```

## Quick Start

Activate the WSL conda environment:

```bash
conda activate chrono-go1
```

View the accepted baseline:

```bash
python view_stand_policy.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 5000 --action-filter-tau 0.05
```

Run a fixed-0.8 diagnostic:

```bash
python diagnose_policy.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/current_baseline_fixed08_smoke
```

Run a compact regression:

```bash
python run_regression.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --name current_baseline_fixed08 --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05
```

## Next Work

Before randomized-friction training, run bridge checks at fixed friction values:

```bash
python friction_curriculum.py bridge-check
```

If the bridge checks pass, continue from the accepted filtered baseline with
fixed `action_filter_tau=0.05`:

```bash
python friction_curriculum.py friction-randomization
```

Acceptance for the next phase still requires settled-window diagnostics, slip
analysis when needed, and viewer checks. Survival alone is not enough.

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

Exact `25/25/25/25` loading is not the natural target for this home pose.
Left/right balance remains a useful diagnostic.
