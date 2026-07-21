# Handoff: V3.1 Position-Motor Standing Baseline

This is the freshest local working-state brief. If it disagrees with runtime
code or diagnostics JSON, trust `go1_env.py`, `project_config.py`, and the
latest diagnostics artifacts first.

## Current State

The active worktree uses standing v3.1:

```text
65D relative-state observation
no absolute world XYZ in policy input
relative-height termination
Chrono position-motor actuator
```

Current checkpoint:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
```

The directory name contains `50k` because it was the planned run folder. The
promoted checkpoint is `5000_steps.zip`; do not describe this model as trained
for 50k steps.

Accepted scope:

```text
absolute world XYZ removed from policy input
relative-height termination
Chrono position-motor action interface restored
50 Hz control / 200 Hz physics timing implemented
action_filter_tau = 0.05 for filtered-control comparisons
home pose per leg = [0.0, 0.7, -1.4]
action scale = 0.20
```

Not accepted yet:

```text
clean-standing acceptance under the new 50 Hz timing
RN1/RN2 reset-noise robustness
push recovery
friction randomization
observation noise
```

The previous V3.1 friction-slice interpretation was wrong. Since clean standing
does not demand meaningful horizontal shear, changing friction values without
pushes does not prove friction robustness. Friction randomization should be
revisited after random push recovery is meaningful.

The 50 Hz control retiming is also a modeling correction: earlier reward and
filter tuning was partly compensating for policy updates that were too fast.
See ADR-020 in `docs/experiments/fixed_friction_standing.md`.

## Previous Clean-Standing Evidence

Clean fixed `mu=0.8`, no reset noise, no pushes, before the 50 Hz control
retiming:

```text
diagnostics/v3p1_fixed08_005k_clean_mu08_confirm30/summary.json
episodes: 30
result: 30/30 nominal
active-reference drift: 0.000207 m
settled total contact foot slip: 0.007678 m
settled contact switches: 0
settled min foot load: 26.95 N
max settled friction usage: 0.03711
max non-foot load: 0.0
```

This was accepted because the old clean-standing failure modes were controlled:
long-hold foot creep, contact shuffling, foot unloading, non-foot support, base
drift, and action-jitter-driven slip.

After switching to 50 Hz control with 200 Hz physics, the old checkpoint is no
longer automatically accepted. Current retiming smoke:

```text
diagnostics/control50hz_retiming_smoke1/summary.json
episodes: 1
result: 1/1 nominal
active-reference drift: 0.000059 m
settled total contact foot slip: 0.238558 m
settled contact switches: 0
settled min foot load: 26.87 N
action_filter_tau: 0.05, alpha: 0.285714
```

## Current Reward And Control

Important constants in `go1_env.py`:

```python
_HOME_JOINT_ANGLES = np.tile([0.0, 0.7, -1.4], 4).astype(np.float32)
_ACTION_SCALE = 0.20
_ALIVE_BONUS = 1.0
_UPRIGHT_REWARD_WEIGHT = 0.15
_POSE_PENALTY_WEIGHT = 0.30
_CONTROL_PENALTY_WEIGHT = 0.03
_ANG_VEL_PENALTY_WEIGHT = 0.01
_XZ_VEL_PENALTY_WEIGHT = 1.00
_JOINT_VEL_PENALTY_WEIGHT = 0.02
_ACTION_RATE_PENALTY_WEIGHT = 0.05
_RAW_ACTION_RATE_PENALTY_WEIGHT = 0.02
_FILTER_LAG_PENALTY_WEIGHT = 0.02
_TILT_PENALTY_WEIGHT = 0.25
_FOOT_CONTACT_MEAN_WEIGHT = 1.00
_FOOT_CONTACT_WORST_WEIGHT = 2.00
_FOOT_SLIP_PENALTY_WEIGHT = 50.00
_FOOT_SLIP_GATE_TOTAL = 0.03
_FOOT_ANCHOR_PENALTY_WEIGHT = 0.10
_BASE_DRIFT_PENALTY_WEIGHT = 0.05
_CONTACT_SWITCH_PENALTY_WEIGHT = 0.10
_ANCHOR_RESET_PENALTY_WEIGHT = 0.50
_ANCHOR_DEACTIVATION_PENALTY_WEIGHT = 1.00
_LOAD_QUALITY_RAMP_STEPS = 50
_STANCE_QUALITY_RAMP_STEPS = 100
_MIN_FOOT_LOAD = 20.0
```

Action interface:

```python
target_q = clip(home_q + 0.20 * executed_action, joint_low, joint_high)
```

The action filter is part of the control interface:

```python
alpha = dt / (tau + dt)
executed_action = previous_executed_action + alpha * (raw_action - previous_executed_action)
```

Timing:

```text
physics timestep = 0.005 s  (200 Hz)
control timestep = 0.020 s  (50 Hz)
physics substeps per policy action = 4
1000 RL steps = 20 s simulated time
```

With `tau=0.05` and control `dt=0.020`, `alpha=0.285714`.

## Decision Table

| Run / model | Key change | Main result | Decision |
|---|---|---|---|
| old fixed/AB baselines | early standing/friction attempts | survival hid drift/slip/contact problems | Historical |
| anchor5 support checkpoint | foot-anchor/support improvement | better support, long-hold creep remained | Rejected as final |
| base drift weight `10.0` | stronger stationarity | worse drift, switching, unloading | Rejected |
| freeze-action diagnostic | freeze action after 1000 steps | slip collapsed; action jitter identified | Accepted diagnostic |
| jitter-suppression 5k | higher action-rate and joint-velocity pressure | drift improved, one-foot creep persisted | Intermediate |
| normalized foot-slip `0.05` | load-weighted slip reward | did not improve failure mode | Rejected |
| stance-shape `0.05`/`0.005` | relative-foot stance reward | failed to beat jitter baseline | Rejected |
| action filter `tau=0.05` | low-pass policy actions | fixed clean standing creep | Accepted control interface |
| v2 filtered/friction/reset lineage | pre-v3 robustness work | accepted in v2, not transferable to v3.1 | Historical accepted |
| v3 relative 35D | removed absolute XYZ | survived but failed slip gate | Rejected recipe |
| v3.1 65D 5k | relative obs, filter-state input, slip-aligned reward | clean standing accepted | Active baseline |
| V3.1 friction slices | fixed friction changes without pushes | not meaningful for robustness | Retracted interpretation |

## Canonical Commands

Compile active code:

```bash
python -m py_compile go1_env.py train_stand.py evaluate_stand.py diagnose_policy.py view_stand_policy.py run_regression.py compare_friction_slices.py friction_curriculum.py analyze_slip_timeline.py diagnostics.py project_config.py view_env.py chrono_go1_soil.py
```

View current baseline:

```bash
python view_stand_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 1000 --action-filter-tau 0.05
```

Run a clean diagnostic:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 1000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --out diagnostics/v3p1_position_motor_clean_mu08_smoke1
```

## Next Step

RN1/RN2 reset-noise ranges are defined but not accepted yet. Test them directly
before training.

```text
RN1: yaw [-pi, pi], roll/pitch +/-0.05 rad, base X/Z +/-0.03 m,
     height +/-0.015 m, linear X/Z +/-0.10 m/s, linear Y +/-0.03 m/s,
     angular X/Z +/-0.15 rad/s, angular Y +/-0.20 rad/s,
     joint pos hip/thigh/knee +/-0.04 / +/-0.08 / +/-0.10 rad,
     joint velocity +/-0.20 rad/s

RN2: yaw [-pi, pi], roll/pitch +/-0.12 rad, base X/Z +/-0.10 m,
     height +/-0.030 m, linear X/Z +/-0.25 m/s, linear Y +/-0.05 m/s,
     angular X/Z +/-0.40 rad/s, angular Y +/-0.50 rad/s,
     joint pos hip/thigh/knee +/-0.10 / +/-0.12 / +/-0.15 rad,
     joint velocity +/-0.50 rad/s
```

Current progression:

```text
1. test and possibly train RN1/RN2 reset recovery
2. random push recovery
3. friction randomization after pushes make friction meaningful
4. observation noise
```

RN3 is a temporary/debug-only deterministic upper-RN2 probe and should not be
documented as accepted.
