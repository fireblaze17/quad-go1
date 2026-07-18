# Training Roadmap

This document records the current decision path for Chrono Go1 standing. Git
history and diagnostics artifacts contain exact old code states; this file
summarizes what is currently believed and what should happen next.

## Current Direction

The active worktree now targets standing v3.1: no absolute world XYZ in the
policy input and no absolute world-height termination. V3.1 fixed `mu=0.8`
clean standing is accepted, including spawn X/Z and ground-height
coordinate-invariance screens. The last accepted v2 result remains important
evidence for friction/reset-noise robustness, but v2 checkpoints are
shape-incompatible with the active 65D v3.1 observation.

Current v3.1 fixed-standing baseline:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
action_filter_tau = 0.05
foot_friction = 2.0
```

Accepted v3.1 fixed-standing evidence:

```text
fixed mu=0.8, clean reset: 30/30 nominal
active-reference drift: 0.000207 m
settled contact foot slip: 0.007678 m
settled contact switches: 0
settled min foot load: 26.95 N

spawn X/Z offsets through +/-0.5 m: all 30/30 nominal
ground-height offsets through +/-0.20 m: all 30/30 nominal
worst coordinate-screen drift: 0.000288 m
worst coordinate-screen slip: 0.008764 m
```

The last accepted v2 randomized-friction flat standing result was accepted over
effective `mu=0.5-1.2` with the filtered-control baseline, and the same
checkpoint was accepted for clean/RN-1/RN-2 reset-state noise:

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

Reset-noise keeper confirmation also passed without extra training:

```text
reset levels: clean, RN-1, RN-2
friction slices: 0.5, 0.8, 1.2
episodes: 100 per condition
failure_type_counts: {'nominal': 100} on every condition
survival_rate: 1.000 on every condition
worst active-reference drift: 0.002813 m
worst settled contact foot slip: 0.003756 m
settled contact switches: 0 on every condition
worst settled min foot load: 28.53 N
max settled friction usage: 0.02035
```

## Active Reward And Control

The active v3.1 reward keeps the accepted standing shape, adds filter-state
visibility in the observation, and aligns reward pressure with the drift/slip
acceptance gates:

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
    - loaded_foot_slip_penalty
    - foot_anchor_penalty
    - base_drift_penalty
    - contact_switch_penalty
    - anchor_reset_penalty
    - anchor_deactivation_penalty
    - raw_action_rate_penalty
    - filter_lag_penalty
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
raw_action_rate         0.02 * mean(raw_action_delta^2)
filter_lag              0.02 * mean((raw_action - executed_action)^2)
tilt_penalty            0.25 * (trunk_x_up^2 + trunk_y_up^2)
angular_vel_penalty     0.01 * mean(trunk_angular_velocity^2)
xz_vel_penalty          1.00 * mean([vx, vz]^2)
foot_contact_penalty    mean missing load + 2.00 * worst missing load
foot_slip_penalty       50.00 * loaded_step_slip / 0.03 m
foot_anchor_penalty     0.10 * normalized anchor excess beyond 0.005 m
base_drift_penalty      0.05 * normalized active-ref drift beyond 0.01 m
contact_switch          0.10 per settled hysteresis switch
anchor_reset            0.50 per reset
anchor_deactivation     1.00 per deactivation
```

The action filter is part of the control interface, not a reward term:

```python
alpha = dt / (tau + dt)
executed_action = previous_executed_action + alpha * (raw_action - previous_executed_action)
```

For the last accepted v2 baseline, `tau=0.05` and `dt=0.002`, so
`alpha=0.038462`.

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
reset-state noise RN-1/RN-2          accepted; passed without training
relative-state standing v3           attempted; first scratch recipe failed Stage 1
relative-state standing v3.1         accepted fixed mu=0.8 + coordinate invariance
```

Important negative lesson: stronger stationarity rewards were less effective
than fixing the control interface. The policy already knew a quiet standing
action; the failure was persistent high-frequency action updates reaching the
contacts.

## Current Diagnostic Workflow

Use filtered-control diagnostics for current v3.1 fixed-standing checks:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 5000 --action-filter-tau 0.05 --log-every-step --out diagnostics/v3p1_fixed08_005k_clean_mu08_timeline
python analyze_slip_timeline.py diagnostics/v3p1_fixed08_005k_clean_mu08_timeline/timeline.csv
python run_regression.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --name v3p1_fixed08_005k_clean_mu08_confirm30 --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 30 --max-steps 5000 --action-filter-tau 0.05
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

The fixed-slice gate already passed for the v2 randomized checkpoint:

```text
checkpoint: runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
slices: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
episodes: 30 per slice
result: nominal on every episode
```

Reset-state noise is accepted through RN-2 for the current v2 baseline. However,
the v2 observation included absolute world coordinates, so observation noise
and terrain work should wait until the v3.1 relative-state checkpoint re-passes
friction and reset-noise gates.

The first v3 implementation removed absolute world XYZ from policy input and
replaced absolute-height termination with relative-height termination. Static
validation passed, but the first scratch fixed-`mu=0.8` training recipe did not
produce a clean checkpoint:

```text
run: runs/stand_v3_relative_obs_fixed08_500k
stopped: about 198k timesteps because live rollout length collapsed
best failed checkpoint: 25k
25k result: survival 30/30, active drift 0.009009 m, settled slip 0.510896 m
gate: failed, because settled slip must be <= 0.03 m
```

Current next sequence:

```text
1. re-check fixed friction slices on v3.1
2. re-check reset-noise gates on v3.1
3. then add observation noise
4. then move to SCM/deformable terrain bridge and locomotion policies
```

Before accepting v3.1 for broader robustness, re-run fixed effective-friction
slices on the v3.1 checkpoint:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 1.2 --friction-max 1.2 --episodes 30 --max-steps 5000 --action-filter-tau 0.05 --out diagnostics/v3p1_fixed08_005k_mu12_confirm30
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

Reset-state noise is implemented as an opt-in environment/CLI feature:

```text
--reset-noise-level clean|rn1|rn2|rn3
--reset-noise-components combined|joint_pos|joint_vel|roll_pitch|base_height|base_velocity
```

RN-1/RN-2 passed screening and keeper confirmation without training. Use the
same 30-episode screening and 100-episode keeper pattern for future robustness
axes. Pushes and actuator/model randomization are deliberately deferred outside
the current project scope.

## Historical Notes

- `docs/experiments/friction_curriculum.md` records the old A/AB/C curriculum.
  Those policies are archived pre-filter evidence, not current baselines.
- `docs/experiments/standing_v2.md` is a historical fixed-friction model card.
- Reset-noise experiments are accepted through RN-2 for v2. Observation noise is
  deferred until v3.1 re-passes friction and reset-noise gates.
- `nominal_load_sanity.py` was removed after its useful load reference was
  documented.
