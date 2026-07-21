# Quad Go1

Project Chrono + PPO reinforcement-learning project for a Unitree Go1-style
quadruped. The current milestone is a reproducible standing controller with
diagnostics strong enough to separate real standing improvements from visual
"it did not fall" checks.

This is also a learning project for Project Chrono and reinforcement learning.
Some earlier interpretations were revised as the modeling assumptions became
clearer; those corrections are kept in the docs because they explain why the
current workflow exists.

## Current Status

The active policy is standing v3.1:

```text
runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
```

The run directory says `50k` because that was the planned training folder. The
promoted checkpoint is `stand_policy_5000_steps.zip`, so the active model has
5000 PPO timesteps in this v3.1 lineage.

Active runtime setup:

```text
actuator: Chrono position motor
home pose per leg: [0.0, 0.7, -1.4]
action scale: 0.20
action_filter_tau: 0.05 for filtered-control comparisons
observation: 65D v3.1 relative-state observation
physics timestep: 0.005 s
control timestep: 0.020 s
control frequency: 50 Hz
physics substeps per policy action: 4
```

Current active-code claim:

```text
no absolute world XYZ in policy input
relative-height termination
position-motor action interface restored
50 Hz control / 200 Hz physics timing implemented
```

Not accepted yet:

```text
clean-standing acceptance under the new 50 Hz timing
RN1/RN2 reset-noise robustness
random push recovery
friction randomization
observation noise
```

Earlier docs treated fixed-friction slices as a friction pass. That was a
mistake: without horizontal disturbances, changing friction does not stress the
standing policy in a meaningful way. Friction testing becomes useful after the
policy is pushed or otherwise asked to create real horizontal shear.

## Clean Standing Evidence

V3.1 was the clean-standing baseline under the earlier one-action-per-physics
step timing. Those diagnostics showed the old standing failure modes were
controlled: contact shuffling, foot creep, action jitter, foot unloading,
non-foot support, and base drift.

Clean fixed `mu=0.8`, no reset noise, no pushes:

```text
episodes: 30
result: 30/30 nominal
active-reference drift: 0.000207 m
settled total contact foot slip: 0.007678 m
settled contact switches: 0
settled min foot load: 26.95 N
max settled friction usage: 0.03711
max non-foot load: 0.0
```

After switching to 50 Hz control with 200 Hz physics, this checkpoint must be
retrained or revalidated before clean standing is accepted again under the new
timing.

The timing change is intentional. Earlier reward/action-filter work was partly
compensating for updating the policy too fast; ADR-020 records the correction to
slower policy control with faster physics integration.

## Quick Start

Activate the supported WSL conda environment:

```bash
conda activate chrono-go1
```

View the active clean-standing baseline:

```bash
python view_stand_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 1000 --action-filter-tau 0.05
```

Run a clean-standing diagnostic:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 1000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --out diagnostics/v3p1_position_motor_clean_mu08_smoke1
```

Inspect one detailed clean-standing timeline:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 1000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --log-every-step --out diagnostics/v3p1_fixed08_005k_clean_mu08_timeline
python analyze_slip_timeline.py diagnostics/v3p1_fixed08_005k_clean_mu08_timeline/timeline.csv
```

## Reward And Control

The active v3.1 observation is 65D and removes absolute world XYZ from policy
input. It includes relative height, trunk orientation/velocity, joint state,
previous executed action, support-relative reference errors, foot loads, and
contact flags.

The policy outputs normalized joint-position offsets:

```python
target_q = clip(home_q + 0.20 * executed_action, joint_low, joint_high)
```

Chrono position motors apply those targets.

Timing:

```text
physics timestep: 0.005 s  (200 Hz)
control timestep: 0.020 s  (50 Hz)
physics substeps per policy action: 4
1000 RL steps = 20 s simulated time
```

The action filter is part of the control interface, not a reward term:

```python
alpha = dt / (tau + dt)
executed_action = previous_executed_action + alpha * (raw_action - previous_executed_action)
```

With `tau=0.05` and control `dt=0.020`, `alpha=0.285714`.

Important active reward settings:

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
contact_switch          0.10 per hysteresis switch
anchor_reset            0.50 per reset
anchor_deactivation     1.00 per deactivation
load quality ramp       first 50 steps
stance quality ramp     first 100 steps
minimum foot load       20 N
```

## Main Issues Encountered

The main lesson is that survival alone is weak evidence. Standing is judged by
settled drift, loaded-foot slip, contact switching, foot load, friction usage,
non-foot contact, action jitter, and observation design.

Key issues:

- Long-hold foot creep after apparently stable standing:
  [ADR-002](docs/experiments/fixed_friction_standing.md#adr-002-anchor5-improved-support-but-failed-long-hold-creep)
- Over-strong base drift reward `10.0` worsening contact behavior:
  [ADR-003](docs/experiments/fixed_friction_standing.md#adr-003-base-drift-100-was-too-strong)
- Action jitter causing slip even when a stable pose existed:
  [ADR-005](docs/experiments/fixed_friction_standing.md#adr-005-freeze-action-diagnostic-identified-action-jitter)
- Rejected normalized foot-slip and stance-shape branches:
  [ADR-007](docs/experiments/fixed_friction_standing.md#adr-007-normalized-foot-slip-005-did-not-help),
  [ADR-008](docs/experiments/fixed_friction_standing.md#adr-008-stance-shape-005-and-0005-were-rejected)
- Absolute world XYZ in the old v2 observation:
  [ADR-015](docs/experiments/fixed_friction_standing.md#adr-015-relative-state-standing-v3-attempt-stopped-at-fixed-mu-gate)
- V3.1 clean-standing recovery with relative observation and slip-aligned reward:
  [ADR-016](docs/experiments/fixed_friction_standing.md#adr-016-v31-filter-state-and-slip-aligned-reward)
- Friction-slice interpretation corrected:
  [ADR-018](docs/experiments/fixed_friction_standing.md#adr-018-friction-slice-claim-retracted-until-pushes-make-friction-meaningful)

## Next Work

Current progression:

```text
1. test and possibly train RN1/RN2 reset recovery
2. random push recovery
3. friction randomization after pushes make friction meaningful
4. observation noise
```

RN3 is temporary/debug-only right now and should not be treated as a formal
accepted reset-noise level.

## Docs Map

- [docs/reproducibility.md](docs/reproducibility.md) - copy-paste commands
- [docs/reproduction_ladder.md](docs/reproduction_ladder.md) - closest reproduction path from untrained policy to current result
- [docs/training_roadmap.md](docs/training_roadmap.md) - current research direction
- [docs/experiments/fixed_friction_standing.md](docs/experiments/fixed_friction_standing.md) - ADR log for standing experiments
- [docs/chrono_port_notes.md](docs/chrono_port_notes.md) - Chrono import, contact, physics, and material decisions
- [docs/collision_debug_log.md](docs/collision_debug_log.md) - collision/contact debugging history
- [HANDOFF.md](HANDOFF.md) - local working-state brief
