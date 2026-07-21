# Reproducibility

Copy-paste commands for the active 48D / 50 Hz / implicit-limited-drive standing worktree. Historical
experiment details live in `docs/experiments/`.

## Static Checks

```bash
conda activate chrono-go1
python -m py_compile go1_env.py train_stand.py diagnose_policy.py view_stand_policy.py view_env.py analyze_slip_timeline.py diagnostics.py project_config.py chrono_go1_soil.py
```

Check the active observation shape:

```bash
python -c "from go1_env import Go1Env; e=Go1Env(max_steps=1); o,_=e.reset(); print(o.shape, e.observation_space.shape); e.close()"
```

Expected:

```text
(48,) (48,)
```

## Active Setup

```text
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
physics substeps per RL step: 4
episode length: 1000 RL steps = 20 s simulated time
```

Active reward:

```text
tracking_lin_vel_zero   +1.0
tracking_ang_vel_zero   +0.5
lin_vel_y               -2.0
ang_vel_xz              -0.05
orientation             -5.0
base_height             -30.0
torques                 -0.0001
dof_acc                 -2.5e-7
action_rate             -0.01
dof_pos_limits          -10.0
collision               -1.0
alive reward            no
command velocity        no
old custom terms        diagnostics only, zero reward weight
```

There is no accepted 48D implicit-limited-drive model yet. Older position-motor and 65D
checkpoints cannot be loaded or promoted directly under this actuator because
the dynamics and observation assumptions changed.

Future locomotion should be reproduced as a flat command-conditioned policy, not
as HRL. Standing remains the zero-command case. The active `48D` observation
already includes `command_vx`, `command_vz`, and `command_yaw_rate`, currently
hardcoded to zero.

## Train Current 48D Implicit-Limited-Drive Setup

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

Training writes:

```text
runs/stand_v4_implicit_limited_drive_reward_aligned_1m/checkpoints/
runs/stand_v4_implicit_limited_drive_reward_aligned_1m/eval_checkpoints/
runs/stand_v4_implicit_limited_drive_reward_aligned_1m/best_model.zip
runs/stand_v4_implicit_limited_drive_reward_aligned_1m/learning_curve.csv
runs/stand_v4_implicit_limited_drive_reward_aligned_1m/learning_curve.json
runs/stand_v4_implicit_limited_drive_reward_aligned_1m/learning_curve.png
```

`best_model.zip` is selected by training-time eval reward. It still needs
`diagnose_policy.py` gate checks before promotion.

## Evaluate New Checkpoints

Clean 30-episode screen:

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

One-episode timeline:

```bash
python diagnose_policy.py runs/stand_v4_implicit_limited_drive_reward_aligned_1m/checkpoints/stand_policy_25000_steps.zip \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --episodes 1 \
  --max-steps 1000 \
  --reset-noise-level clean \
  --reset-noise-components combined \
  --log-every-step \
  --out diagnostics/v4_implicit_limited_drive_025k_timeline

python analyze_slip_timeline.py diagnostics/v4_implicit_limited_drive_025k_timeline/timeline.csv
```

Viewer:

```bash
python view_stand_policy.py runs/stand_v4_implicit_limited_drive_reward_aligned_1m/checkpoints/stand_policy_25000_steps.zip \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000
```

Zero-action viewer:

```bash
python view_env.py \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000 \
  --reset-noise-level clean \
  --reset-noise-components combined
```

## Reset-Noise Screens

RN1/RN2 are implemented but not accepted. Use them only after clean 45D standing
passes.

```bash
for LEVEL in rn1 rn2; do
  python diagnose_policy.py runs/stand_v4_implicit_limited_drive_reward_aligned_1m/checkpoints/stand_policy_25000_steps.zip \
    --terrain flat \
    --friction-min 0.8 \
    --friction-max 0.8 \
    --episodes 30 \
    --max-steps 1000 \
    --reset-noise-level "$LEVEL" \
    --reset-noise-components combined \
    --out "diagnostics/v4_implicit_limited_drive_reset_${LEVEL}_mu08_screen30"
done
```

Component screen if combined RN2 fails:

```bash
for COMPONENT in joint_pos joint_vel roll_pitch yaw base_height base_position base_velocity; do
  python diagnose_policy.py runs/stand_v4_implicit_limited_drive_reward_aligned_1m/checkpoints/stand_policy_25000_steps.zip \
    --terrain flat \
    --friction-min 0.8 \
    --friction-max 0.8 \
    --episodes 30 \
    --max-steps 1000 \
    --reset-noise-level rn2 \
    --reset-noise-components "$COMPONENT" \
    --out "diagnostics/v4_implicit_limited_drive_rn2_${COMPONENT}_mu08_screen30"
done
```

`runs/` artifacts are gitignored and must be supplied out of band.
