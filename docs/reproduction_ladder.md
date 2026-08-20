# Reproduction Ladder

This ladder records the model lineage used for the project result: train the flat NSC actuator-net policy, then fine-tune that policy on SCM.

Model zips are not tracked by git. A fresh checkout needs these artifacts placed at:

```text
runs/default_baseline/checkpoints/flat_150m_baseline.zip
runs/default_baseline/checkpoints/default_baseline.zip
```

## Stage 1: Flat NSC Baseline

The flat baseline artifact is:

```text
runs/default_baseline/checkpoints/flat_150m_baseline.zip
```

It is a neutral copy of:

```text
runs/go1_nvidia_flat_reward_dr_actuator_net_v1_200m/checkpoints/stand_policy_149997600_steps.zip
```

Training configuration:

```text
terrain: flat
friction: 0.8
actuator_model: actuator_net
action_clip: 100
num_envs: 24
n_steps: 256
batch_size: 1536
n_epochs: 3
learning_rate: 1e-4
learning_rate_final: 1e-4
clip_range: 0.1
target_kl: 0.015
ent_coef: 0.001
gamma: 0.99
gae_lambda: 0.95
max_grad_norm: 1.0
checkpoint_freq: 1000000
target_total_steps: 200000000
selected checkpoint: 149997600 steps
```

Equivalent reproduction command:

```bash
python train_stand.py \
  --save-dir runs/flat_nsc_baseline_repro \
  --target-total-steps 200000000 \
  --max-steps 1000 \
  --actuator-model actuator_net \
  --num-envs 24 \
  --n-steps 256 \
  --batch-size 1536 \
  --n-epochs 3 \
  --learning-rate 0.0001 \
  --learning-rate-final 0.0001 \
  --clip-range 0.1 \
  --target-kl 0.015 \
  --ent-coef 0.001 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --max-grad-norm 1.0 \
  --checkpoint-freq 1000000
```

View the flat baseline on NSC:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/flat_150m_baseline.zip
```

## Stage 2: SCM Fine-Tune

The promoted project baseline is:

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

It is a neutral copy of:

```text
runs/default_scm_finetune_lr3e5_clip005_from516m_v1_100m_continue/checkpoints/stand_policy_570990864_steps.zip
```

Fine-tuning configuration:

```text
env_backend: scm
actuator_model: actuator_net
num_envs: 24
n_steps: 256
batch_size: 1536
n_epochs: 3
learning_rate: 3e-5
learning_rate_final: 3e-5
clip_range: 0.05
target_kl: 0.001
ent_coef: 0.001
gamma: 0.99
gae_lambda: 0.95
max_grad_norm: 0.5
checkpoint_freq: 1000000
selected checkpoint: 570990864 steps
```

Equivalent fine-tuning command:

```bash
python train_stand.py \
  --env-backend scm \
  --save-dir runs/default_scm_finetune_repro \
  --resume-model runs/default_baseline/checkpoints/flat_150m_baseline.zip \
  --target-total-steps 570990864 \
  --max-steps 1000 \
  --actuator-model actuator_net \
  --num-envs 24 \
  --n-steps 256 \
  --batch-size 1536 \
  --n-epochs 3 \
  --learning-rate 0.00003 \
  --learning-rate-final 0.00003 \
  --clip-range 0.05 \
  --target-kl 0.001 \
  --ent-coef 0.001 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --max-grad-norm 0.5 \
  --checkpoint-freq 1000000
```

## Verification Commands

View the promoted baseline on NSC:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/default_baseline.zip
```

View the promoted baseline on SCM:

```bash
python view_scm_policy_vsg.py \
  runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0 \
  --max-steps 500 \
  --render-fps 1000000 \
  --ignore-termination
```

Run a fixed-command NSC diagnostic:

```bash
python diagnose_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0 \
  --episodes 1 \
  --max-steps 1000 \
  --out diagnostics/default_forward_eval \
  --log-every-step
```

Run a fixed-command SCM diagnostic:

```bash
python diagnose_policy.py \
  --env-backend scm \
  runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0 \
  --episodes 1 \
  --max-steps 1000 \
  --out diagnostics/default_scm_forward_eval \
  --log-every-step
```
