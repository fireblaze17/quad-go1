# Reproducibility

This file gives copy-paste commands for the current Chrono Go1 standing
workflow.

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
  runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip

required control setup:
  --action-filter-tau 0.05
```

The accepted fixed-0.8 confirmation was:

```text
episodes: 30
failure_type_counts: {'nominal': 30}
survival_rate: 1.000
settled_base_displacement_from_active_ref: 0.001637 m
settled_total_contact_foot_slip_distance: 0.003516 m
settled_total_contact_switches: 0
settled_min_foot_load: 26.68 N
```

## View

```bash
python view_stand_policy.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 5000 --action-filter-tau 0.05
```

## Evaluate

```bash
python evaluate_stand.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10 --max-steps 5000 --action-filter-tau 0.05
```

## Diagnose

One-episode timeline:

```bash
python diagnose_policy.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/current_baseline_fixed08_timeline
python analyze_slip_timeline.py diagnostics/current_baseline_fixed08_timeline/timeline.csv
```

Thirty-episode confirmation:

```bash
python diagnose_policy.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/current_baseline_fixed08_confirm30
```

Regression wrapper:

```bash
python run_regression.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --name current_baseline_fixed08_confirm30 --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05
```

## Friction Bridge Checks

Before randomized-friction training, run fixed-friction slices around the
accepted baseline:

```bash
python friction_curriculum.py bridge-check
```

Equivalent manual pattern for one slice:

```bash
python diagnose_policy.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --terrain flat --friction-min 0.7 --friction-max 0.7 --episodes 10 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/friction_bridge_filtered2k_mu_0p7
```

Compare against a candidate once one exists:

```bash
python compare_friction_slices.py --baseline runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --candidate CANDIDATE.zip --name candidate_vs_filtered2k_slices --episodes 10 --max-steps 5000 --mu 0.6 0.7 0.8 0.9 1.0 --action-filter-tau 0.05
```

## Training Continuation

The accepted baseline was produced by a conservative 5k fine-tune with the
action filter enabled. Future standing continuations should keep the filter:

```bash
python train_stand.py \
  --load runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip \
  --save-dir runs/stand_friction_randomized_tau005_from_filtered2k \
  --terrain flat \
  --friction-min 0.6 \
  --friction-max 1.0 \
  --max-steps 5000 \
  --timesteps 50000 \
  --checkpoint-freq 10000 \
  --learning-rate 0.00005 \
  --clip-range 0.05 \
  --target-kl 0.01 \
  --action-filter-tau 0.05
```

Use shorter bridge tests before committing to long randomized-friction runs.

## Checkpoint Availability

`runs/` is intentionally gitignored. Model artifacts live out of band. Keep
each run's `args.json` and `env_constants.json` with its checkpoint.

Important local paths:

```text
runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip  # current baseline
runs/stand_jitter_suppression_from_anchor5_10k/checkpoints/stand_policy_5000_steps.zip    # previous baseline
runs/stand_fixed_clean_contact2_anchor5_from25k_10k/checkpoints/stand_policy_5000_steps.zip # archived support checkpoint
runs/stand_friction_ab_065_095/final_model.zip                                           # archived pre-filter friction baseline
```

## Smoke Test

```bash
python diagnose_policy.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/doc_refresh_smoke_tau005
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
