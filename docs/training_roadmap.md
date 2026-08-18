# Training Roadmap

## Current Goal

Keep the codebase centered on the default locomotion stack and improve new runs from the current baseline.

Baseline:

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

## Default Training Setup

```text
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
```

## Metrics To Watch

- `rollout/ep_len_mean`
- `rollout/ep_rew_mean`
- `reward_terms/tracking_lin_vel_reward_mean`
- `reward_terms/tracking_ang_vel_reward_mean`
- `reward_terms/flat_orientation_l2_reward_mean`
- `reward_terms/action_rate_reward_mean`
- `reward_terms/torques_reward_mean`
- `policy/std_mean`
- `policy/mean_action_rms`
- `policy/sampled_action_rms`
- `feet/contact_fraction_*`
- `feet/contact_switches_*_mean_per_env`

## Interpretation

- Rising reward with stable or rising episode length is healthy.
- Falling policy standard deviation with stable reward usually means the deterministic policy is catching up to the sampled policy.
- If velocity tracking rises but contact metrics get worse, inspect the viewer before extending the run.
- If reward improves while episode length falls, run fixed-command diagnostics to see whether the policy is exploiting short bursts before termination.

Continue a run when reward, velocity tracking, or episode length are still improving and contact behavior is not degrading. Stop or branch a run when reward and velocity tracking are flat over many checkpoints, policy standard deviation has mostly settled, and deterministic fixed-command viewers match the intended behavior.

Before promoting a model, run fixed forward, fixed lateral, fixed yaw, and sampled-command viewers. Use diagnostics to confirm velocity tracking, torque saturation, action-rate penalty, contact switching, and foot-load behavior.

## Default Training Command

```bash
python train_stand.py \
  --save-dir runs/default_new_run \
  --target-total-steps 100000000 \
  --checkpoint-freq 1000000
```

TensorBoard:

```bash
tensorboard --logdir runs/default_new_run/tensorboard --port 6006
```

## SCM Fine-Tuning Path

CRM/SPH was tested, but it is too computationally heavy for practical fine-tuning on this setup. SCM is the active deformable-terrain backend.

Start SCM experiments from the current baseline and keep the policy interface unchanged:

```bash
python train_stand.py \
  --env-backend scm \
  --save-dir runs/default_scm_finetune_v1 \
  --resume-model runs/default_baseline/checkpoints/default_baseline.zip \
  --target-total-steps 110000000 \
  --checkpoint-freq 1000000
```

Use the VSG viewer for qualitative SCM checks:

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

## SCM Fork Experiments

The current long SCM fine-tune line reached:

```text
runs/default_scm_finetune_to501m_v1_continue/checkpoints/stand_policy_454992720_steps.zip
```

A separate optimizer-aggressiveness fork starts from that checkpoint and changes only the PPO update size:

```text
run: runs/default_scm_finetune_lr3e5_clip005_from455m_v1_100m_fork
resume checkpoint: runs/default_scm_finetune_to501m_v1_continue/checkpoints/stand_policy_454992720_steps.zip
target total steps: 554992720
learning_rate: 3e-5
learning_rate_final: 3e-5
clip_range: 0.05
checkpoint_freq: 1000000
```

Everything else should stay aligned with the current SCM fine-tune recipe unless the experiment explicitly says otherwise. The training script writes the run arguments to `args.json` and PPO/TensorBoard scalars under the run's `tensorboard/` directory, so this fork can be compared against the previous `1e-5`, `clip_range=0.02` line.
