# Training Reference

This document records the training settings used for the project result. The active model lineage is documented in `docs/reproduction_ladder.md`.

## Flat NSC Baseline Settings

Model:

```text
runs/default_baseline/checkpoints/flat_150m_baseline.zip
```

Training settings:

```text
terrain: flat
friction: 0.8
actuator_model: actuator_net
action_clip: 100
num_envs: 24
n_steps: 256
rollout_size: 6144
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

Reproduction command:

```bash
python train_policy.py \
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

## SCM Fine-Tune Settings

Model:

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

Training settings:

```text
env_backend: scm
actuator_model: actuator_net
num_envs: 24
n_steps: 256
rollout_size: 6144
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

Fine-tuning command:

```bash
python train_policy.py \
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

## TensorBoard Metrics

Primary training metrics:

```text
rollout/ep_len_mean
rollout/ep_rew_mean
train/approx_kl
train/clip_fraction
train/entropy_loss
train/explained_variance
train/value_loss
policy/std_mean
policy/mean_action_rms
policy/sampled_action_rms
```

Reward and behavior metrics:

```text
reward_terms/tracking_lin_vel_reward_mean
reward_terms/tracking_ang_vel_reward_mean
reward_terms/flat_orientation_l2_reward_mean
reward_terms/action_rate_reward_mean
reward_terms/torques_reward_mean
feet/contact_fraction_*
feet/contact_switches_*_mean_per_env
```

Open TensorBoard:

```bash
tensorboard --logdir runs/default_new_run/tensorboard --port 6006
```
