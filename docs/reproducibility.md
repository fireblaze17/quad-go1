# Reproducibility

This file gives copy-paste commands for the Chrono Go1 standing workflow.
The active worktree now uses the v3.1 65D relative-state observation, so old v2
checkpoints are shape-incompatible with active code. V3.1 is accepted for fixed
`mu=0.8` clean standing plus coordinate-invariance screens; v2 remains the last
accepted friction/reset-noise robust result.

For the full command lineage from an untrained policy to the current v3.1
branch, see [reproduction_ladder.md](reproduction_ladder.md).

## Platform

```text
OS tested:         WSL Ubuntu on Windows
Python:            3.12
PyChrono:          10.0.0
Gymnasium:         1.2.3
Stable-Baselines3: 2.8.0
Torch:             2.11.0
World frame:       Chrono Y-up
Robot model:       models/go1/go1_chrono.urdf
```

Activate the environment:

```bash
conda activate chrono-go1
```

On Ankus's WSL machine, the explicit interpreter is:

```text
/home/ankus/miniforge3/envs/chrono-go1/bin/python
```

## Static Checks

```bash
python -m py_compile go1_env.py train_stand.py evaluate_stand.py diagnose_policy.py view_stand_policy.py run_regression.py compare_friction_slices.py friction_curriculum.py analyze_slip_timeline.py diagnostics.py project_config.py view_env.py chrono_go1_soil.py
python train_stand.py --help
python evaluate_stand.py --help
python diagnose_policy.py --help
python run_regression.py --help
python compare_friction_slices.py --help
python friction_curriculum.py --help
```

Runtime reset-noise support is intentionally present and defaults to clean/off:

```bash
python train_stand.py --help | rg "reset-noise"
python diagnose_policy.py --help | rg "reset-noise"
```

Expected result: both commands show `--reset-noise-level` and
`--reset-noise-components`.

## Active V3.1 Fixed-Standing Baseline

```text
checkpoint:
  runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip

required control setup:
  --action-filter-tau 0.05
  foot_friction = 2.0
```

Accepted v3.1 fixed-standing confirmation:

```text
fixed mu=0.8, clean reset
episodes: 30
failure_type_counts: {'nominal': 30}
survival_rate: 1.000
settled_base_displacement_from_active_ref: 0.000207 m
settled_total_contact_foot_slip_distance: 0.007678 m
settled_total_contact_switches: 0
settled_min_foot_load: 26.95 N
```

Accepted coordinate-invariance screens:

```text
spawn X/Z offsets through +/-0.5 m: all 30/30 nominal
ground-height offsets through +/-0.20 m: all 30/30 nominal
worst coordinate-screen drift: 0.000288 m
worst coordinate-screen slip: 0.008764 m
```

View:

```bash
python view_stand_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 5000 --action-filter-tau 0.05
```

Diagnose:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --out diagnostics/v3p1_fixed08_005k_clean_mu08_confirm30
```

Coordinate check:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --spawn-x 0.5 --spawn-z 0.5 --out diagnostics/v3p1_fixed08_005k_spawn_x0p5_z0p5_confirm30
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --ground-height-offset 0.2 --out diagnostics/v3p1_fixed08_005k_ground_h0p20_confirm30
```

## Last Accepted V2 Friction/Reset Baseline

```text
checkpoint:
  runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip

required control setup:
  --action-filter-tau 0.05
  foot_friction = 2.0
```

The accepted effective `0.5-1.2` confirmation was:

```text
slices: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
episodes: 30 per slice
failure_type_counts: {'nominal': 30} on every slice
survival_rate: 1.000 on every slice
worst settled_base_displacement_from_active_ref: 0.001558 m
worst settled_total_contact_foot_slip_distance: 0.003871 m
settled_total_contact_switches: 0 on every slice
worst settled_min_foot_load: 28.53 N
```

The accepted reset-noise confirmation was:

```text
reset levels: clean, RN-1, RN-2
friction slices: 0.5, 0.8, 1.2
episodes: 100 per condition
failure_type_counts: {'nominal': 100} on every condition
survival_rate: 1.000 on every condition
worst settled_base_displacement_from_active_ref: 0.002813 m
worst settled_total_contact_foot_slip_distance: 0.003756 m
settled_total_contact_switches: 0 on every condition
worst settled_min_foot_load: 28.53 N
```

The strongest single reset-noise evidence file is:

```text
diagnostics/keeper_reset_rn2_mu_0p5_confirm100/summary.json
```

## V2 View

```bash
python view_stand_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 0.5 --friction-max 1.2 --max-steps 5000 --action-filter-tau 0.05
```

## V2 Evaluate

```bash
python evaluate_stand.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 0.5 --friction-max 1.2 --episodes 10 --max-steps 5000 --action-filter-tau 0.05
```

## V2 Diagnose

One-episode timeline:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/v2_robust_mu12_timeline
python analyze_slip_timeline.py diagnostics/v2_robust_mu12_timeline/timeline.csv
```

Thirty-episode confirmation:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/v2_robust_mu12_confirm30
```

Regression wrapper:

```bash
python run_regression.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --name v2_robust_mu12_confirm30 --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05
```

## Friction Slice Checks

Chrono combines ground and foot friction by `min(ground, foot)` in this SMC
setup. Foot friction is intentionally set to `2.0` as a cap-removal setting, so
target ground values through `mu=1.2` are the effective friction values under
test.

Run one fixed slice:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/friction_050_120_cap2_current10k_mu_1p2
```

Run the full accepted fixed-slice set:

```bash
for MU in 0.5 0.6 0.8 0.9 1.0 1.1 1.2; do
  SAFE_MU=${MU/./p}
  python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min "$MU" --friction-max "$MU" --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --out "diagnostics/friction_050_120_cap2_current10k_mu_${SAFE_MU}"
done
```

Timeline for a failing slice:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/v2_robust_mu12_timeline
python analyze_slip_timeline.py diagnostics/v2_robust_mu12_timeline/timeline.csv
```

## Fallback Training

Training is not needed for the accepted `0.5-1.2` baseline. If a future
material or reward change breaks any fixed slice, restart from the clean
filtered fixed fallback with lower LR:

```bash
python train_stand.py \
  --load runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip \
  --save-dir runs/stand_friction_random_050_120_tau005_from_filtered2k_lowlr \
  --terrain flat \
  --friction-min 0.5 \
  --friction-max 1.2 \
  --max-steps 5000 \
  --timesteps 20000 \
  --checkpoint-freq 2000 \
  --learning-rate 0.000025 \
  --clip-range 0.05 \
  --target-kl 0.005 \
  --action-filter-tau 0.05
```

Evaluate each checkpoint on fixed slices `0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2`.
Select by worst-slice behavior, not randomized average reward.

Checkpoint evaluation pattern:

```bash
for CKPT in runs/stand_friction_random_050_120_tau005_from_filtered2k_lowlr/checkpoints/stand_policy_*_steps.zip; do
  STEM=$(basename "$CKPT" .zip)
  for MU in 0.5 0.6 0.8 0.9 1.0 1.1 1.2; do
    SAFE_MU=${MU/./p}
    python diagnose_policy.py "$CKPT" --terrain flat --friction-min "$MU" --friction-max "$MU" --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --out "diagnostics/${STEM}_mu_${SAFE_MU}"
  done
done
```

## Reset-Noise Evaluation

Reset noise is accepted through RN-2. Defaults are clean/off, so old commands
remain valid unless reset-noise flags are supplied.

Debug one RN-2 timeline at fixed `mu=0.8`:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level rn2 --reset-noise-components combined --log-every-step --out diagnostics/reset_noise_rn2_mu08_timeline
```

Component ablation screening at `mu=0.8`:

```bash
for COMPONENT in joint_pos joint_vel roll_pitch base_height base_velocity; do
  python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level rn2 --reset-noise-components "$COMPONENT" --out "diagnostics/reset_noise_rn2_${COMPONENT}_mu08_screen30"
done
```

Combined reset/friction screen:

```bash
for LEVEL in clean rn1 rn2; do
  for MU in 0.5 0.8 1.2; do
    SAFE_MU=${MU/./p}
    python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min "$MU" --friction-max "$MU" --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level "$LEVEL" --reset-noise-components combined --out "diagnostics/reset_noise_${LEVEL}_mu_${SAFE_MU}_screen30"
  done
done
```

Fallback RN-2 training command only if future changes regress the accepted
keeper grid:

```bash
python train_stand.py \
  --load runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip \
  --save-dir runs/stand_reset_noise_rn2_tau005_from_friction10k_100k \
  --terrain flat \
  --friction-min 0.5 \
  --friction-max 1.2 \
  --max-steps 5000 \
  --timesteps 100000 \
  --checkpoint-freq 10000 \
  --learning-rate 0.000025 \
  --clip-range 0.05 \
  --target-kl 0.005 \
  --action-filter-tau 0.05 \
  --reset-noise-level rn2 \
  --reset-noise-components combined
```

Keeper confirmation pattern:

```bash
for LEVEL in clean rn1 rn2; do
  for MU in 0.5 0.8 1.2; do
    SAFE_MU=${MU/./p}
    python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min "$MU" --friction-max "$MU" --episodes 100 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level "$LEVEL" --reset-noise-components combined --out "diagnostics/keeper_reset_${LEVEL}_mu_${SAFE_MU}_confirm100"
  done
done
```

That keeper grid passed for the v2 robustness baseline, so the fallback training
command was not run.

## Checkpoint Availability

`runs/` is intentionally gitignored. Model artifacts live out of band. Keep
each run's `args.json` and `env_constants.json` with its checkpoint.

Important local paths:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip    # active v3.1 fixed-standing baseline
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip # last accepted v2 friction/reset baseline
runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip  # clean filtered fixed fallback
runs/stand_jitter_suppression_from_anchor5_10k/checkpoints/stand_policy_5000_steps.zip    # previous baseline
runs/stand_fixed_clean_contact2_anchor5_from25k_10k/checkpoints/stand_policy_5000_steps.zip # archived support checkpoint
runs/stand_friction_ab_065_095/final_model.zip                                           # archived pre-filter friction baseline
```

## Smoke Test

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/doc_refresh_v3p1_smoke_mu08_tau005
```

## Nominal Load Reference

The zero-action home-pose reference at fixed `mu=0.8`:

```text
FR: 27.17%
FL: 27.55%
RR: 22.46%
RL: 22.82%
front/rear: 54.7/45.3
left/right: 50.4/49.6
```

Exact `25/25/25/25` foot loading is not the natural target for the current home
pose. Left/right balance remains a valid diagnostic.
