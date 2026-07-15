# Training Roadmap

This document records the current decision path for Chrono Go1 standing. Git
history and diagnostics artifacts contain exact old code states; this file
summarizes what is currently believed and what should happen next.

## Current Direction

Randomized-friction flat standing is accepted over effective `mu=0.5-1.2` with
the filtered-control baseline:

```text
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
action_filter_tau = 0.05
foot_friction = 2.0
```

The accepted 30-episode, 5000-step fixed-slice confirmation:

```text
slices: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
failure_type_counts: {'nominal': 30} on every slice
survival_rate: 1.000 on every slice
worst active-reference drift: 0.001558 m
worst settled contact foot slip: 0.003871 m
settled contact switches: 0 on every slice
worst settled min foot load: 28.53 N
max settled friction usage: 0.01833
```

No extra training was needed after raising foot friction to `2.0`; the current
randomized 10k checkpoint already passed the full effective `0.5-1.2` gate.

## Active Reward And Control

The active reward keeps the accepted standing shape, plus settled foot-anchor
and base-reference diagnostics:

```python
reward = (
    alive_bonus
    + upright_reward
    - tilt_penalty
    - pose_penalty
    - control_penalty
    - joint_vel_penalty
    - action_rate_penalty
    - ang_vel_penalty
    - xz_vel_penalty
    - foot_contact_penalty
    - foot_anchor_penalty
    - base_drift_penalty
)
```

Current weights:

```text
alive_bonus             1.00
upright_reward          0.15 * upright_score
pose_penalty            0.30 * mean(joint_error^2)
control_penalty         0.03 * mean(executed_action^2)
joint_vel_penalty       0.02 * mean(joint_velocity^2)
action_rate_penalty     0.05 * mean(executed_action_delta^2)
tilt_penalty            0.25 * (trunk_x_up^2 + trunk_y_up^2)
angular_vel_penalty     0.01 * mean(trunk_angular_velocity^2)
xz_vel_penalty          1.00 * mean([vx, vz]^2)
foot_contact_penalty    2.00 * mean(missing_foot_load^2)
foot_slip_penalty       0.00
foot_anchor_penalty     5.00 * planted anchor error after step 100
base_drift_penalty      2.00 * max(0, active_ref_drift - 0.01)^2 after step 100
```

The action filter is part of the control interface, not a reward term:

```python
alpha = dt / (tau + dt)
executed_action = previous_executed_action + alpha * (raw_action - previous_executed_action)
```

For the accepted baseline, `tau=0.05` and `dt=0.002`, so `alpha=0.038462`.

## Decision History

The detailed ADR-style log is in
[experiments/fixed_friction_standing.md](experiments/fixed_friction_standing.md).
The closest command-by-command reproduction path is in
[reproduction_ladder.md](reproduction_ladder.md).
The short version:

```text
old fixed/AB baselines              archived; survival was not clean standing
anchor5 support checkpoint          useful, but failed long-hold creep
base drift weight 10.0              rejected
freeze-action diagnostic            accepted; identified action jitter
jitter-suppression fine-tune         accepted as intermediate baseline
normalized foot slip 0.05           rejected
stance-shape 0.05 and 0.005         rejected
eval-only action filter sweep        accepted; tau=0.05 smallest clean tau
filtered fine-tune                   accepted; 2k checkpoint promoted
friction randomization 0.6-0.9       accepted; 10k checkpoint promoted
foot friction 2.0 cap removal        accepted; 0.5-1.2 passed without training
```

Important negative lesson: stronger stationarity rewards were less effective
than fixing the control interface. The policy already knew a quiet standing
action; the failure was persistent high-frequency action updates reaching the
contacts.

## Current Diagnostic Workflow

Use filtered-control diagnostics for all current baseline checks:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/current_baseline_mu12_timeline
python analyze_slip_timeline.py diagnostics/current_baseline_mu12_timeline/timeline.csv
python run_regression.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --name current_baseline_mu12_confirm30 --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05
```

Acceptance signals:

```text
survival_rate
failure_type_counts
settled_base_displacement_from_active_ref
settled_total_contact_foot_slip_distance
settled_total_contact_switches
settled_min_foot_load
settled foot-load shares
foot-anchor displacement/reset/deactivation diagnostics
viewer drift/slip/contact behavior
```

## Next Experiment

Chrono's default SMC material composition combines two contact frictions with
`min(ground, foot)`. The active foot friction is now intentionally set to `2.0`
as a cap-removal setting, not a measured Go1 value. For target ground values
through `mu=1.2`, effective friction equals ground friction.

The fixed-slice gate already passed for the current randomized checkpoint:

```text
checkpoint: runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
slices: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
episodes: 30 per slice
result: nominal on every episode
```

Future work should add the next robustness axis one at a time, starting from
this checkpoint and keeping `action_filter_tau=0.05`:

```text
1. reset-state noise
2. observation noise
3. terrain variation
4. actuator/model randomization
```

Before accepting any new checkpoint, re-run fixed effective-friction slices:

```bash
python diagnose_policy.py runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/current_baseline_mu12_confirm30
```

Recommended acceptance:

```text
each tested mu reaches max_steps
active-reference drift <= 0.03 m
settled contact switches = 0 preferred
settled min foot load near or above 20 N
settled loaded-foot slip remains low
friction_usage stays comfortably below 1.0
no repeated foot_slip / one_foot_creep / all_feet_creep classification
viewer shows no obvious sliding
```

Promote future checkpoints by worst fixed-slice behavior, not by average
randomized reward. Stop or reject a run if fixed `mu=0.8` regresses, contact
switches return, loaded-foot slip grows, or friction usage repeatedly
approaches/exceeds `1.0`.

## Historical Notes

- `docs/experiments/friction_curriculum.md` records the old A/AB/C curriculum.
  Those policies are archived pre-filter evidence, not current baselines.
- `docs/experiments/standing_v2.md` is a historical fixed-friction model card.
- Reset-noise experiments are the next likely robustness axis, but they should
  start from the current randomized 10k baseline and be accepted only after
  fixed-slice checks remain clean.
- `nominal_load_sanity.py` was removed after its useful load reference was
  documented.
