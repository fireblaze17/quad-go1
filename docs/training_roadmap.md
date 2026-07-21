# Training Roadmap

This roadmap tracks the active standing-policy path. Historical branches remain
documented in `docs/experiments/`.

## Current Baseline

```text
policy:
  runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip

runtime:
  Chrono position motors
  home pose per leg = [0.0, 0.7, -1.4]
  action scale = 0.20
  action_filter_tau = 0.05 for filtered-control comparisons
  physics timestep = 0.005 s
  control timestep = 0.020 s
  control frequency = 50 Hz
  physics substeps per policy action = 4
  observation = 65D relative-state v3.1
```

Active-code status:

```text
no absolute world XYZ in policy input
relative-height termination
Chrono position-motor action interface
50 Hz control / 200 Hz physics timing
```

Not accepted:

```text
clean standing under the new 50 Hz timing
RN1/RN2 reset-noise robustness
push recovery
friction randomization
observation noise
```

Previous clean-standing evidence before the 50 Hz control retiming:

```text
diagnostics/v3p1_fixed08_005k_clean_mu08_confirm30/summary.json
30/30 nominal
active-reference drift = 0.000207 m
settled loaded-foot slip = 0.007678 m
settled contact switches = 0
settled min foot load = 26.95 N
max non-foot load = 0.0
```

The old checkpoint must be retrained or revalidated before clean standing is
accepted under the new timing.

ADR-020 documents the timing lesson: smoothness should come first from a sane
simulation/control split, not only from reward penalties and action filtering.

## Corrected Friction Interpretation

Fixed-friction slices without pushes are not a robustness pass for this project.
The standing pose does not create enough horizontal shear for friction changes
to be meaningful. Friction randomization should happen after random push
recovery exists.

## Active Work Order

1. Re-establish clean V3.1 standing under 50 Hz control / 200 Hz physics.
2. Define and test RN1/RN2 reset-noise behavior under the current v3.1
   position-motor baseline.
3. If RN1/RN2 fails, diagnose component failures before training.
4. Add random push recovery and compare policy behavior against zero-action
   standing.
5. Revisit friction randomization only after pushes make friction matter.
6. Add observation noise after reset and push recovery are understood.

RN3 remains debug/stretch only and is not an accepted reset-noise level.

## Canonical Commands

View the active baseline:

```bash
python view_stand_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 1000 --action-filter-tau 0.05
```

Run a clean diagnostic:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 1 --max-steps 1000 --action-filter-tau 0.05 --reset-noise-level clean --reset-noise-components combined --out diagnostics/v3p1_position_motor_clean_mu08_smoke1
```

Train a new v3.1 position-motor continuation/from-scratch run:

```bash
python train_stand.py \
  --save-dir runs/stand_v3p1_position_motor \
  --terrain flat \
  --friction-min 0.8 \
  --friction-max 0.8 \
  --max-steps 1000 \
  --timesteps 1000000 \
  --checkpoint-freq 50000 \
  --learning-rate 0.0003 \
  --clip-range 0.2 \
  --eval-during-training \
  --eval-freq 50000 \
  --eval-episodes 5 \
  --early-stop-patience 5 \
  --early-stop-min-delta 1.0
```
