# Reproduction Ladder

This guide records the closest reproducible path from an untrained policy to
the last accepted v2 standing controller, plus the active v3/v3.1 relative-state
retraining branch. The run artifacts under `runs/` are gitignored, so exact
reproduction requires preserving or sharing checkpoints out of band.

Last accepted v2 endpoint:

```text
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
Go1Env(action_filter_tau=0.05)
_FOOT_FRICTION = 2.0
effective flat friction mu = 0.5-1.2
reset noise clean/RN-1/RN-2
```

The v2 endpoint requires the pre-v3 37D observation code. It is
shape-incompatible with the active 65D v3.1 worktree. See ADR-015 and ADR-016
in `docs/experiments/fixed_friction_standing.md` for the relative-state branch.

## Environment Setup

Supported runtime:

```bash
conda activate chrono-go1
python -m py_compile go1_env.py train_stand.py evaluate_stand.py diagnose_policy.py view_stand_policy.py run_regression.py compare_friction_slices.py friction_curriculum.py analyze_slip_timeline.py diagnostics.py project_config.py view_env.py chrono_go1_soil.py
```

The project uses WSL Ubuntu for PyChrono and WSLg for Irrlicht viewers.

## Active V3.1 Reward

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
- raw_action_rate_penalty
- filter_lag_penalty
- angular_velocity_penalty
- raw X/Z velocity penalty
- missing-foot-load contact penalty
- loaded-foot slip penalty
- foot-anchor penalty after the stance ramp
- base-drift penalty after the stance ramp
- contact-switch and anchor reset/deactivation penalties
```

Current weights:

```text
alive_bonus             1.00
upright                 0.15
pose                    0.30
control                 0.03
joint_velocity          0.02
action_rate             0.05
raw_action_rate         0.02
filter_lag              0.02
tilt                    0.25
angular_velocity        0.01
xz_velocity             1.00
foot_contact            mean 1.00, worst-foot 2.00
foot_slip               50.00 * loaded_step_slip / 0.03 m
foot_anchor             0.10 normalized beyond 0.005 m
base_drift              0.05 normalized beyond 0.01 m
contact_switch          0.10
anchor_reset            0.50
anchor_deactivation     1.00
minimum_foot_load       20 N
load_quality_ramp       50 steps
stance_quality_ramp     100 steps
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

## Stage 7: Reset-State Noise Evaluation

Reset noise was added after friction robustness, using the existing randomized
10k checkpoint. No PPO training was needed.

Implemented reset levels:

```text
RN-1: tiny height, roll/pitch, joint position, joint velocity, and base velocity perturbations
RN-2: target perturbations, roughly double RN-1 for pose and larger for velocities
RN-3: stretch level for future testing, not part of the accepted gate
```

Screening used 30 episodes per condition. Keeper confirmation used:

```text
policy: runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
reset levels: clean, RN-1, RN-2
friction slices: 0.5, 0.8, 1.2
episodes: 100 per condition
max_steps: 5000
action_filter_tau: 0.05
```

Representative command:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 0.5 --friction-max 0.5 --episodes 100 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level rn2 --reset-noise-components combined --out diagnostics/keeper_reset_rn2_mu_0p5_confirm100
```

Strongest single evidence file:

```text
diagnostics/keeper_reset_rn2_mu_0p5_confirm100/summary.json
```

Result:

```text
100/100 nominal on every clean/RN-1/RN-2 x mu=0.5/0.8/1.2 condition
worst active-reference drift: 0.002813 m
worst settled total contact foot slip: 0.003756 m
contact switches: 0 on every condition
worst settled min foot load: 28.53 N
max settled friction usage: 0.02035
max non-foot load: 0.0
```

Decision: accept the current checkpoint as reset-noise capable through RN-2
without additional training.

## Stage 8: V3/V3.1 Relative-State Branch

The v2 result worked, but it used absolute world base position in the policy
observation and absolute world height in termination. V3 removes that dependency
before observation noise, SCM terrain, and locomotion.

V3 design rule:

```text
no absolute world X/Y/Z in policy input
no absolute world-height termination
```

First v3 attempt:

```bash
python train_stand.py \
  --save-dir runs/stand_v3_relative_obs_fixed08_500k \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 5000 \
  --timesteps 500000 \
  --checkpoint-freq 25000 \
  --learning-rate 0.0003 \
  --clip-range 0.2 \
  --action-filter-tau 0.05
```

Result: rejected. The best early checkpoint survived but slid:

```text
runs/stand_v3_relative_obs_fixed08_500k/checkpoints/stand_policy_25000_steps.zip
30/30 nominal
active-reference drift: 0.009009 m
settled slip: 0.510896 m
gate: failed, slip must be <= 0.03 m
```

V3.1 corrected the mismatch by expanding the observation to 65D and adding
slip-aligned normalized penalties. It still excludes absolute world XYZ.

V3.1 training command:

```bash
python train_stand.py \
  --save-dir runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 5000 \
  --timesteps 50000 \
  --checkpoint-freq 5000 \
  --learning-rate 0.0003 \
  --clip-range 0.2 \
  --action-filter-tau 0.05
```

Promoted v3.1 checkpoint:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
```

Fixed `mu=0.8` confirmation:

```text
30/30 nominal
active-reference drift: 0.000207 m
settled total contact foot slip: 0.007678 m
contact switches: 0
settled min foot load: 26.95 N
max non-foot load: 0.0
```

Coordinate-invariance confirmation:

```text
spawn X/Z offsets through +/-0.5 m: all 30/30 nominal
ground-height offsets through +/-0.20 m: all 30/30 nominal
worst coordinate-screen drift: 0.000288 m
worst coordinate-screen slip: 0.008764 m
```

## Stage 9: V3.1 Effective Friction 0.5-1.2

V3.1 was evaluated first instead of immediately fine-tuned. The v2 project
history showed that extra PPO training can degrade standing, so the gate was:
train only if fixed friction slices fail.

Source checkpoint:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
```

One reward implementation cleanup was made before evaluation: the remaining
hard settled-quality early return was removed. The same ramped scales are used
instead:

```text
load quality ramp:   clip(step / 50, 0, 1)
stance quality ramp: clip(step / 100, 0, 1)
```

Reference-dependent anchor/base penalties remain zero before the step-100
standing reference exists; direct loaded-foot slip and contact-switch terms
ramp smoothly from the start.

Screening command:

```bash
for MU in 0.5 0.6 0.8 0.9 1.0 1.1 1.2; do
  SAFE_MU=${MU/./p}
  python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min "$MU" --friction-max "$MU" --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --out "diagnostics/v3p1_friction_bridge_mu_${SAFE_MU}_screen30"
done
```

Keeper command:

```bash
for MU in 0.5 0.6 0.8 0.9 1.0 1.1 1.2; do
  SAFE_MU=${MU/./p}
  python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min "$MU" --friction-max "$MU" --episodes 100 --max-steps 5000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --out "diagnostics/v3p1_friction_keeper_mu_${SAFE_MU}_confirm100"
done
```

Result:

```text
100/100 nominal on every fixed slice
slices: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
worst active-reference drift: 0.000215 m
worst settled total contact foot slip: 0.007615 m
contact switches: 0 on every slice
worst settled min foot load: 26.95 N
max settled friction usage: 0.03424
max non-foot load: 0.0
```

Per-slice keeper table:

```text
mu   episodes  nominal  drift_m   slip_m    switches  min_load_N  max_friction_usage
0.5  100       100/100   0.000201  0.007565  0         26.946      0.034239
0.6  100       100/100   0.000200  0.007615  0         26.951      0.026631
0.8  100       100/100   0.000215  0.007392  0         26.977      0.025404
0.9  100       100/100   0.000215  0.007392  0         26.977      0.022581
1.0  100       100/100   0.000215  0.007392  0         26.977      0.020323
1.1  100       100/100   0.000215  0.007392  0         26.977      0.018476
1.2  100       100/100   0.000215  0.007392  0         26.977      0.016936
```

Decision: accept the v3.1 5k checkpoint as effective-friction robust through
`mu=0.5-1.2` without additional PPO training. Next reproduction step is to
repeat Stage 7 reset-noise screens with this v3.1 checkpoint before adding
observation noise.

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
