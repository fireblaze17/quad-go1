# Reproducibility

This file gives copy-paste commands for the current Chrono Go1 standing
workflow.

For the full command lineage from an untrained policy to the current checkpoint,
see [reproduction_ladder.md](reproduction_ladder.md).

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

Runtime reset-noise support should be absent:

```bash
rg "reset-noise|reset_noise|RN-A|rn_a|RESET_NOISE" go1_env.py train_stand.py evaluate_stand.py diagnose_policy.py view_stand_policy.py run_regression.py compare_friction_slices.py project_config.py friction_curriculum.py
```

Expected result: no runtime-code hits.

## Current Baseline

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

## View

```bash
python view_stand_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 0.5 --friction-max 1.2 --max-steps 5000 --action-filter-tau 0.05
```

## Evaluate

```bash
python evaluate_stand.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 0.5 --friction-max 1.2 --episodes 10 --max-steps 5000 --action-filter-tau 0.05
```

## Diagnose

One-episode timeline:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/current_baseline_mu12_timeline
python analyze_slip_timeline.py diagnostics/current_baseline_mu12_timeline/timeline.csv
```

Thirty-episode confirmation:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/current_baseline_mu12_confirm30
```

Regression wrapper:

```bash
python run_regression.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --name current_baseline_mu12_confirm30 --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05
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
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/current_baseline_mu12_timeline
python analyze_slip_timeline.py diagnostics/current_baseline_mu12_timeline/timeline.csv
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

## Checkpoint Availability

`runs/` is intentionally gitignored. Model artifacts live out of band. Keep
each run's `args.json` and `env_constants.json` with its checkpoint.

Important local paths:

```text
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip # current baseline
runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip  # clean filtered fixed fallback
runs/stand_jitter_suppression_from_anchor5_10k/checkpoints/stand_policy_5000_steps.zip    # previous baseline
runs/stand_fixed_clean_contact2_anchor5_from25k_10k/checkpoints/stand_policy_5000_steps.zip # archived support checkpoint
runs/stand_friction_ab_065_095/final_model.zip                                           # archived pre-filter friction baseline
```

## Smoke Test

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --friction-min 1.2 --friction-max 1.2 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/doc_refresh_smoke_mu12_tau005
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
