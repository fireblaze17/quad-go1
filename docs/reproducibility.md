# Reproducibility

This file contains copy-paste commands for the active v3.1 position-motor
baseline. Historical experiment details live in `docs/experiments/`.

## Environment Check

```bash
conda activate chrono-go1
python -m py_compile go1_env.py train_stand.py evaluate_stand.py diagnose_policy.py view_stand_policy.py view_env.py run_regression.py compare_friction_slices.py friction_curriculum.py analyze_slip_timeline.py diagnostics.py project_config.py chrono_go1_soil.py
```

## Active Baseline

```text
policy:
  runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip

runtime:
  Chrono position motors
  home pose per leg = [0.0, 0.7, -1.4]
  action scale = 0.20
  action_filter_tau = 0.05
  physics timestep = 0.005 s
  control timestep = 0.020 s
  control frequency = 50 Hz
  physics substeps per RL step = 4
  max_steps = 1000 for current active checks, or 20 s simulated time
```

Active-code scope:

```text
no absolute world XYZ in policy input
relative-height termination
Chrono position-motor action interface
50 Hz control / 200 Hz physics timing
```

Not accepted:

```text
clean standing under the new 50 Hz timing
RN1/RN2 reset noise
push recovery
friction randomization
observation noise
```

The old V3.1 clean-standing checkpoint is useful as a source checkpoint, but it
was trained before the 50 Hz control retiming. Reproduce old evidence only as
historical context; retrain or revalidate before accepting the new timing.

## View Baseline

```bash
python view_stand_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000 \
  --action-filter-tau 0.05
```

Zero-action environment viewer:

```bash
python view_env.py \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000 \
  --reset-noise-level clean \
  --reset-noise-components combined
```

## Clean Diagnostic

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --episodes 1 \
  --max-steps 1000 \
  --action-filter-tau 0.05 \
  --reset-noise-level clean \
  --reset-noise-components combined \
  --out diagnostics/v3p1_position_motor_clean_mu08_smoke1
```

Timeline:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --episodes 1 \
  --max-steps 1000 \
  --action-filter-tau 0.05 \
  --reset-noise-level clean \
  --reset-noise-components combined \
  --log-every-step \
  --out diagnostics/v3p1_position_motor_clean_mu08_timeline

python analyze_slip_timeline.py diagnostics/v3p1_position_motor_clean_mu08_timeline/timeline.csv
```

## Reset-Noise Screens

RN1/RN2 ranges are implemented but not accepted. Use these only as screens.

```bash
for LEVEL in rn1 rn2; do
  python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip \
    --terrain flat \
    --friction-min 0.8 \
    --friction-max 0.8 \
    --episodes 30 \
    --max-steps 1000 \
    --action-filter-tau 0.05 \
    --reset-noise-level "$LEVEL" \
    --reset-noise-components combined \
    --out "diagnostics/v3p1_position_motor_reset_${LEVEL}_mu08_screen30"
done
```

Component screen if combined RN2 fails:

```bash
for COMPONENT in joint_pos joint_vel roll_pitch yaw base_height base_position base_velocity; do
  python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip \
    --terrain flat \
    --friction-min 0.8 \
    --friction-max 0.8 \
    --episodes 30 \
    --max-steps 1000 \
    --action-filter-tau 0.05 \
    --reset-noise-level rn2 \
    --reset-noise-components "$COMPONENT" \
    --out "diagnostics/v3p1_position_motor_rn2_${COMPONENT}_mu08_screen30"
done
```

## Training Template

```bash
python train_stand.py \
  --save-dir runs/stand_v3p1_position_motor \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000 \
  --timesteps 1000000 \
  --checkpoint-freq 50000 \
  --learning-rate 0.0003 \
  --clip-range 0.2 \
  --eval-during-training \
  --eval-freq 50000 \
  --eval-episodes 5 \
  --early-stop-patience 5 \
  --early-stop-min-delta 1.0
```

`runs/` artifacts are gitignored and must be supplied out of band.
