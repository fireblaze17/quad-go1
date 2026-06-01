# Reproducibility

This file gives the commands needed to run the current friction-only standing
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
python -m py_compile go1_env.py train_stand.py evaluate_stand.py diagnose_policy.py view_stand_policy.py run_regression.py compare_friction_slices.py nominal_load_sanity.py project_config.py
python train_stand.py --help
python evaluate_stand.py --help
python diagnose_policy.py --help
python run_regression.py --help
python compare_friction_slices.py --help
```

Runtime reset-noise support should be absent:

```bash
rg "reset-noise|reset_noise|RN-A|rn_a|RESET_NOISE" go1_env.py train_stand.py evaluate_stand.py diagnose_policy.py view_stand_policy.py run_regression.py compare_friction_slices.py nominal_load_sanity.py project_config.py
```

Expected result: no runtime-code hits.

## Checkpoint Availability

`runs/` is intentionally gitignored. Accepted checkpoints are local artifacts
and are not stored in GitHub.

Current kept checkpoints:

```text
runs/stand/final_model.zip
runs/stand_base_v2/final_model.zip
runs/stand_friction_a_07_09/final_model.zip
runs/stand_friction_a_07_09_300k/final_model.zip
runs/stand_friction_ab_065_095/final_model.zip
runs/accepted_backups/
runs/stand_reset_noise_a_slip0005_fullc_from50k_25k/  # archive only
```

Keep each run's `args.json` and `env_constants.json` with `final_model.zip`.

## Baseline Commands

Zero-action environment check:

```bash
python view_env.py
```

Train fixed-friction standing:

```bash
python train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000 --seed 1 --save-dir runs/stand
```

Evaluate:

```bash
python evaluate_stand.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10
```

View:

```bash
python view_stand_policy.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8
```

## PPO Training Knobs

`train_stand.py` exposes optional PPO controls. Defaults preserve normal SB3
behavior:

```bash
--learning-rate
--clip-range
--target-kl
```

These work for both fresh training and `--load` continuation.

## Archived AB Evaluation

AB is the archived old baseline:

```text
runs/stand_friction_ab_065_095/final_model.zip
```

Evaluate on C range:

```bash
python evaluate_stand.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.5 --friction-max 1.1 --episodes 100
```

Run full regression:

```bash
python run_regression.py runs/stand_friction_ab_065_095/final_model.zip --name ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100
```

Fixed-friction slices:

```bash
python compare_friction_slices.py --baseline runs/stand_friction_ab_065_095/final_model.zip --candidate runs/stand_friction_ab_065_095/final_model.zip --name ab_self_slices --episodes 100
```

## Cleaner AB Retry

The next planned training branch starts from friction A:

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

Regress each checkpoint before deciding whether to continue:

```bash
python run_regression.py runs/stand_friction_ab_clean_retry/checkpoints/stand_policy_50000_steps.zip --name ab_clean_retry_50k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_ab_clean_retry/checkpoints/stand_policy_100000_steps.zip --name ab_clean_retry_100k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_ab_clean_retry/checkpoints/stand_policy_150000_steps.zip --name ab_clean_retry_150k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_ab_clean_retry/checkpoints/stand_policy_200000_steps.zip --name ab_clean_retry_200k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_ab_clean_retry/checkpoints/stand_policy_250000_steps.zip --name ab_clean_retry_250k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_ab_clean_retry/final_model.zip --name ab_clean_retry_300k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
```

Compare fixed-friction slices against archived AB:

```bash
python compare_friction_slices.py --baseline runs/stand_friction_ab_065_095/final_model.zip --candidate runs/stand_friction_ab_clean_retry/final_model.zip --name ab_clean_retry_slices --episodes 100
```

View:

```bash
python view_stand_policy.py runs/stand_friction_ab_clean_retry/final_model.zip --terrain flat --friction-min 0.5 --friction-max 1.1
```

## Smoke Tests

Small regression:

```bash
python run_regression.py runs/stand_friction_ab_065_095/final_model.zip --name smoke_ab_clean_friction --friction-min 0.5 --friction-max 1.1 --episodes 2 --max-steps 100
```

Small slice check:

```bash
python compare_friction_slices.py --baseline runs/stand_friction_ab_065_095/final_model.zip --candidate runs/stand_friction_ab_065_095/final_model.zip --name smoke_friction_slices --episodes 2 --max-steps 100 --mu 0.5 0.8
```

## Acceptance Rule

A standing candidate is accepted only if it improves the archived AB behavior:

```text
100/100 survival on C range
better settled drift/slip metrics than AB
healthy min foot load
no persistent viewer sliding or tilt
no non-foot collision load
no obvious biased support mode
```

Reset-noise work should resume only after this friction-only standing baseline
is cleaner.
