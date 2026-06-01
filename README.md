# Quad Go1

Project Chrono robotics simulation and ML project for a Unitree Go1-style
quadruped.

## Goal

1. Build a stable Go1 simulation in Chrono.
2. Wrap it as a Gymnasium environment.
3. Train clean standing on flat randomized friction.
4. Transfer to Chrono SCM deformable terrain.
5. Collect rollouts, train a world model, and add hierarchical skill selection.

## Docs Map

- [README.md](README.md) - current project overview
- [docs/reproducibility.md](docs/reproducibility.md) - exact commands and checks
- [docs/training_roadmap.md](docs/training_roadmap.md) - reward decisions and next work
- [docs/experiments/friction_curriculum.md](docs/experiments/friction_curriculum.md) - friction A/B/C history
- [docs/experiments/standing_v2.md](docs/experiments/standing_v2.md) - fixed-friction model card
- [docs/collision_debug_log.md](docs/collision_debug_log.md) - collision/contact debugging trail
- [docs/chrono_port_notes.md](docs/chrono_port_notes.md) - Chrono import, solver, and contact notes

## Current Status

```text
Stage: randomized-friction standing cleanup

Archived old baseline:
  checkpoint: runs/stand_friction_ab_065_095/final_model.zip
  trained on: friction 0.65-0.95
  accepted on old criteria: B range 0.6-1.0, stress C range 0.5-1.1

Current interpretation:
  AB is still the best archived standing checkpoint, but it is not final-clean.
  New settled-window diagnostics showed drift/slip even before reset-noise work.
  Reset-noise experiments are paused until randomized-friction standing is cleaner.

Active code path:
  friction-only standing
  no reset-noise runtime API
  old AB-era reward restored
  diagnostic tooling kept for settled-window and friction-slice checks

Next work:
  restart before AB, likely from runs/stand_friction_a_07_09/final_model.zip
  train a cleaner AB replacement on randomized friction
  use settled diagnostics, fixed-friction slices, and viewer checks before acceptance
```

## Kept Checkpoints

`runs/` is gitignored and model artifacts live out of band.

```text
runs/stand/final_model.zip
runs/stand_base_v2/final_model.zip
runs/stand_friction_a_07_09/final_model.zip
runs/stand_friction_a_07_09_300k/final_model.zip
runs/stand_friction_ab_065_095/final_model.zip
runs/accepted_backups/
runs/stand_reset_noise_a_slip0005_fullc_from50k_25k/  # archived reference only
```

The reset-noise `slip25` run is kept only as historical evidence. The cleaned
runtime does not support reset-noise evaluation right now.

## Current Reward

The active environment is back to the old friction-era reward shape:

```text
alive_bonus:      1.00
upright:          0.15
pose:             0.30
control:          0.03
joint_velocity:   0.01
action_rate:      0.03
tilt:             0.25
angular_velocity: 0.01
xz_velocity:      0.20
foot_contact:     0.10
```

Removed from active reward/code: reset-noise profiles, clean-standing bonus,
direct load-balance reward experiments, foot-slip reward experiments,
contact-switch reward pressure, and reset-noise comparison modes.

## Project Shape

```text
go1_env.py                 Chrono Gymnasium environment
view_env.py                zero-action/live test harness
train_stand.py             PPO standing-policy training
evaluate_stand.py          headless policy evaluation
view_stand_policy.py       trained-policy viewer
friction_curriculum.py     flat randomized-friction curriculum helper
project_config.py          shared paths and runtime defaults
diagnose_policy.py         headless settled-window diagnosis
run_regression.py          eval + diagnosis regression runner
compare_friction_slices.py fixed-friction AB-vs-candidate comparison
nominal_load_sanity.py     zero-action/home-pose load sanity check
models/go1/go1_chrono.urdf Chrono-specific Go1 URDF
chrono_go1_soil.py         SCM deformable terrain milestone
mujoco/                    MuJoCo baseline, historical reference only
docs/                      decision logs and roadmap
```

## Quick Start

The active development environment is WSL Ubuntu with the `chrono-go1` conda
environment:

```bash
conda activate chrono-go1
```

After activation, use `python`.

```bash
# View zero-action environment
python view_env.py

# Train fixed-friction standing
python train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000 --seed 1 --save-dir runs/stand

# Evaluate a policy
python evaluate_stand.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10

# View a policy
python view_stand_policy.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8
```

## Current Comparison Workflow

Survival is required, but it is not enough. A candidate must also improve
settled drift, contact quality, foot load, and viewer behavior.

```bash
# One friction range: evaluation + diagnosis
python run_regression.py POLICY.zip --name RUN_NAME --friction-min 0.5 --friction-max 1.1 --episodes 100

# Fixed-friction slices against archived AB
python compare_friction_slices.py --baseline runs/stand_friction_ab_065_095/final_model.zip --candidate POLICY.zip --name RUN_NAME_slices --episodes 100

# Nominal home-pose load sanity
python nominal_load_sanity.py --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10

# Viewer check
python view_stand_policy.py POLICY.zip --terrain flat --friction-min 0.5 --friction-max 1.1
```

## Next Training Direction

The next branch should rebuild a cleaner randomized-friction AB-style policy
before returning to reset-noise:

```bash
python train_stand.py \
  --terrain flat \
  --friction-min 0.65 \
  --friction-max 0.95 \
  --load runs/stand_friction_a_07_09/final_model.zip \
  --save-dir runs/stand_friction_ab_clean_retry \
  --timesteps 300000 \
  --seed 1 \
  --checkpoint-freq 50000
```

Then compare it against archived AB:

```bash
python run_regression.py runs/stand_friction_ab_clean_retry/final_model.zip --name ab_clean_retry_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python compare_friction_slices.py --baseline runs/stand_friction_ab_065_095/final_model.zip --candidate runs/stand_friction_ab_clean_retry/final_model.zip --name ab_clean_retry_slices --episodes 100
```

## Metrics Glossary

```text
survival_rate              fraction of episodes reaching max steps
mean_abs_xz_vel            average horizontal trunk velocity magnitude
settled_mean_abs_xz_vel    same metric in the final settled window
base_xz_displacement       final/settled trunk drift from reset position
foot_contact_error         missing-load/contact quality penalty signal
min_foot_load              lowest per-foot vertical contact load
dominant_loaded_leg        foot carrying most load most often
least_loaded_leg           foot carrying least load most often
contact_switch_count       per-foot contact toggles in a window
foot_slip_distance         contact-conditioned tangential foot motion
```

## Notes

- AB passed the old B/C survival and visual checks, but settled diagnostics now
  show it still creeps/slips enough that it should not be treated as final.
- Reset-noise work exposed the problem more clearly; it did not create the root
  issue.
- The codebase is intentionally back to friction-only training so the standing
  controller can be fixed at the source before adding robustness knobs again.
