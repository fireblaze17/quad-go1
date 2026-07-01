# Training Roadmap

This document records the current decision path for Chrono Go1 standing. Git
history and diagnostics artifacts contain exact old code states; this file
summarizes what is currently believed and what should happen next.

## Current Direction

Fixed-friction flat standing at `mu=0.8` is accepted with the filtered-control
baseline:

```text
runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip
action_filter_tau = 0.05
```

The accepted 30-episode, 5000-step fixed-0.8 confirmation:

```text
failure_type_counts: {'nominal': 30}
survival_rate: 1.000
active-reference drift: 0.001637 m
settled total contact foot slip: 0.003516 m
settled total contact switches: 0
settled min foot load: 26.68 N
```

Next work is not another fixed-0.8 reward change. The next step is bridge
testing at nearby fixed frictions, then randomized-friction fine-tuning if the
bridge checks remain clean.

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
```

Important negative lesson: stronger stationarity rewards were less effective
than fixing the control interface. The policy already knew a quiet standing
action; the failure was persistent high-frequency action updates reaching the
contacts.

## Current Diagnostic Workflow

Use filtered-control diagnostics for all current baseline checks:

```bash
python diagnose_policy.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/current_baseline_fixed08_timeline
python analyze_slip_timeline.py diagnostics/current_baseline_fixed08_timeline/timeline.csv
python run_regression.py runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip --name current_baseline_fixed08_confirm30 --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05
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

Run fixed-friction bridge checks before randomized training:

```bash
python friction_curriculum.py bridge-check
```

Recommended bridge acceptance:

```text
each tested mu reaches max_steps
active-reference drift <= 0.03 m
settled contact switches = 0 preferred
settled min foot load near or above 20 N
no repeated foot_slip / one_foot_creep / all_feet_creep classification
viewer shows no obvious sliding
```

If bridge checks pass, continue with conservative PPO from the filtered
baseline:

```bash
python friction_curriculum.py friction-randomization
```

Keep `--action-filter-tau 0.05` during continuation. Treat it as part of the
actuator/control stack unless an explicit ablation is being run.

## Historical Notes

- `docs/experiments/friction_curriculum.md` records the old A/AB/C curriculum.
  Those policies are archived pre-filter evidence, not current baselines.
- `docs/experiments/standing_v2.md` is a historical fixed-friction model card.
- Reset-noise experiments remain archived evidence only. They should not resume
  until filtered standing is tested across friction.
- `nominal_load_sanity.py` was removed after its useful load reference was
  documented.
