# Reproduction Ladder

This guide records the closest reproducible path from an untrained policy to
the current accepted standing controller. The run artifacts under `runs/` are
gitignored, so exact reproduction requires preserving or sharing checkpoints
out of band.

Current accepted endpoint:

```text
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
Go1Env(action_filter_tau=0.05)
_FOOT_FRICTION = 2.0
effective flat friction mu = 0.5-1.2
```

## Environment Setup

Supported runtime:

```bash
conda activate chrono-go1
python -m py_compile go1_env.py train_stand.py evaluate_stand.py diagnose_policy.py view_stand_policy.py run_regression.py compare_friction_slices.py friction_curriculum.py analyze_slip_timeline.py diagnostics.py project_config.py view_env.py chrono_go1_soil.py
```

The project uses WSL Ubuntu for PyChrono and WSLg for Irrlicht viewers.

## Active Reward

The current active reward is:

```text
reward =
  alive_bonus
+ upright_reward
- tilt_penalty
- pose_penalty
- control_penalty
- joint_velocity_penalty
- action_rate_penalty
- angular_velocity_penalty
- raw X/Z velocity penalty
- missing-foot-load contact penalty
- foot-anchor penalty after step 100
- base-drift penalty after step 100
```

Current weights:

```text
alive_bonus             1.00
upright                 0.15
pose                    0.30
control                 0.03
joint_velocity          0.02
action_rate             0.05
tilt                    0.25
angular_velocity        0.01
xz_velocity             1.00
foot_contact            2.00
foot_slip               0.00
foot_anchor             5.00
foot_anchor_deadband    0.005 m
base_drift              2.00
base_drift_deadband     0.01 m
minimum_foot_load       20 N
standing_quality_start  step 100
```

The action filter is not a reward:

```python
alpha = dt / (tau + dt)
executed_action = previous_executed_action + alpha * (raw_action - previous_executed_action)
```

Accepted baseline uses `tau=0.05`.

## Stage 1: Historical Standing V2 From Scratch

This was the original untrained-policy standing run. It is historical and not
the final-clean baseline, but it is the first reproducible training anchor.

```bash
python train_stand.py \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --timesteps 500000 \
  --seed 1 \
  --save-dir runs/stand
```

Historical reward differences from current code:

```text
joint_velocity: 0.01
action_rate:    0.03
xz_velocity:    0.20
foot_contact:   0.10
no settled foot-anchor/base-drift cleanup yet
foot_friction:  0.9
```

Result: survived fixed `mu=0.8` short evaluations, but later settled-window
diagnostics exposed drift, foot slip, and contact-quality issues.

## Stage 2: Support And Anchor Cleanup

The next useful lineage produced:

```text
runs/stand_fixed_clean_contact2_anchor5_from25k_10k/checkpoints/stand_policy_5000_steps.zip
```

Important accepted changes:

```text
foot_contact weight raised to 2.00
foot_anchor weight 5.00 added after step 100
base_drift weight 2.00 added after step 100
raw xz_velocity weight restored to 1.00
minimum foot load kept at 20 N
```

This checkpoint improved support and reduced unloading, but 5000-step
diagnostics still showed long-hold creep.

Rejected branch:

```text
base_drift = 10.0
```

Reason rejected: more drift pressure worsened contact switching and unloading.

## Stage 3: Identify And Suppress Action Jitter

Freeze-action diagnosis showed the standing pose itself was viable: freezing
actions after step 1000 reduced long-hold slip to the millimeter scale. That
identified ongoing policy action jitter as the main failure mode.

Jitter-suppression fine-tune:

```bash
python train_stand.py \
  --load runs/stand_fixed_clean_contact2_anchor5_from25k_10k/checkpoints/stand_policy_5000_steps.zip \
  --save-dir runs/stand_jitter_suppression_from_anchor5_10k \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 5000 \
  --timesteps 10000 \
  --checkpoint-freq 5000 \
  --learning-rate 0.00005 \
  --clip-range 0.05 \
  --target-kl 0.01
```

Reward changes:

```text
action_rate:    0.03 -> 0.05
joint_velocity: 0.01 -> 0.02
```

Promoted intermediate:

```text
runs/stand_jitter_suppression_from_anchor5_10k/checkpoints/stand_policy_5000_steps.zip
```

Result: base drift improved, but one-foot creep remained.

Rejected branches from this stage:

```text
normalized load-weighted foot-slip penalty, weight 0.05
stance-shape penalty, weights 0.05 and 0.005
```

## Stage 4: Action Filter Fine-Tune

An eval-only sweep showed `tau=0.05` was the smallest clean low-pass action
filter. The filter was then moved into `Go1Env` and used during fine-tuning.

```bash
python train_stand.py \
  --load runs/stand_jitter_suppression_from_anchor5_10k/checkpoints/stand_policy_5000_steps.zip \
  --save-dir runs/stand_action_filter_tau005_from_jitter5k_5k \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 5000 \
  --timesteps 5000 \
  --checkpoint-freq 1000 \
  --learning-rate 0.00005 \
  --clip-range 0.05 \
  --target-kl 0.01 \
  --action-filter-tau 0.05
```

Promoted fixed-friction checkpoint:

```text
runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip
```

Fixed `mu=0.8` confirmation:

```text
30/30 nominal
active-reference drift: 0.001637 m
settled total contact foot slip: 0.003516 m
contact switches: 0
settled min foot load: 26.68 N
```

## Stage 5: Randomized Friction 0.6-0.9

After fixed-slice bridge checks, train a conservative randomized-friction
continuation from the filtered fixed checkpoint.

```bash
python train_stand.py \
  --load runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip \
  --save-dir runs/stand_friction_random_060_090_tau005_from_filtered2k \
  --terrain flat \
  --friction-min 0.6 \
  --friction-max 0.9 \
  --max-steps 5000 \
  --timesteps 50000 \
  --checkpoint-freq 5000 \
  --learning-rate 0.00005 \
  --clip-range 0.05 \
  --target-kl 0.01 \
  --action-filter-tau 0.05
```

Promoted checkpoint:

```text
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
```

Reason: the 10k checkpoint had the best worst-slice behavior. Later checkpoints
were not automatically better and were treated as degraded for this selection.

## Stage 6: Extend Effective Friction To 0.5-1.2

Chrono combines ground and foot material friction with `min(ground, foot)`.
The previous `foot_friction=0.9` capped all higher ground values at effective
`0.9`. The current code sets:

```text
_FOOT_FRICTION = 2.0
```

This is a cap-removal modeling choice, not a measured Unitree Go1 material.
With foot friction `2.0`, ground friction slices through `1.2` are the
effective friction values.

No training was run for this stage. The randomized 10k checkpoint was evaluated
directly:

```text
fixed slices: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
episodes: 30 per slice
max_steps: 5000
action_filter_tau: 0.05
```

Result:

```text
30/30 nominal on every slice
worst active-reference drift: 0.001558 m
worst settled total contact foot slip: 0.003871 m
contact switches: 0 on every slice
worst settled min foot load: 28.53 N
max settled friction usage: 0.01833
```

## Acceptance Pattern

For future continuations, select checkpoints by worst fixed-slice behavior:

```text
survival_rate = 1.0
failure_type_counts nominal-only
active-reference drift <= 0.03 m
settled contact switches = 0 preferred
settled min foot load near/above 20 N
settled loaded-foot slip remains low
friction_usage comfortably below 1.0
viewer shows no sliding, chatter, or load collapse
```

Do not promote a checkpoint just because average randomized reward improves.
