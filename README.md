# Quad Go1

Project Chrono reinforcement-learning project for a Unitree Go1-style
quadruped. The current milestone is a reproducible flat-ground standing policy
that remains stable across randomized effective friction.

## Current Result

```text
accepted baseline:
  runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip

required runtime setup:
  action_filter_tau = 0.05
  foot_friction = 2.0

accepted effective friction range:
  mu = 0.5-1.2
```

The checkpoint alone is not the full controller. Reproducing the accepted
behavior requires the environment action filter and the `2.0` foot-friction
cap-removal setting in `go1_env.py`.

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

## Quick Start

Activate the supported WSL conda environment:

```bash
conda activate chrono-go1
```

View the accepted policy:

```bash
python view_stand_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 0.5 --friction-max 1.2 --max-steps 5000 --action-filter-tau 0.05
```

Run one diagnostic episode at the hardest accepted high-friction slice:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/current_baseline_mu12_timeline
python analyze_slip_timeline.py diagnostics/current_baseline_mu12_timeline/timeline.csv
```

Run a compact 30-episode regression:

```bash
python run_regression.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --name current_baseline_mu12_confirm30 --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05
```

## Reward And Control

The active standing reward rewards survival/uprightness and penalizes tilt,
pose error, control effort, joint velocity, action rate, angular velocity, raw
X/Z base velocity, missing foot load, post-settle foot-anchor drift, and
post-settle base drift.

Important active settings:

```text
alive_bonus             1.00
upright                 0.15
pose                    0.30
control                 0.03
joint_velocity          0.02
action_rate             0.05
tilt                    0.25
angular_velocity        0.01
xz_velocity             1.00
foot_contact            2.00
foot_slip               0.00
foot_anchor             5.00, 0.005 m deadband
base_drift              2.00, 0.01 m deadband
standing quality start  step 100
minimum foot load       20 N
```

The action filter is part of the control interface, not a reward term:

```python
alpha = dt / (tau + dt)
executed_action = previous_executed_action + alpha * (raw_action - previous_executed_action)
```

For the accepted baseline, `tau=0.05` and `dt=0.002`, so `alpha=0.038462`.

Rejected reward paths are documented in the ADR log. Stance-shape, normalized
X/Z velocity, base drift `10.0`, and global foot-slip reward `0.05` are not
active.

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

Friction robustness is accepted for flat terrain over effective `mu=0.5-1.2`.
The next research phase should add one robustness axis at a time:

```text
1. reset-state noise
2. observation noise
3. terrain variation
4. actuator/model randomization
5. SCM deformable terrain
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
