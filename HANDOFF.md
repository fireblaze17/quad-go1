# Handoff

## Current Baseline

The current baseline is:

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

It is the active locomotion baseline and should be the first model used for viewing, diagnostics, and comparison.
It currently contains the 571M-step SCM-fine-tuned policy.

Model zips are ignored by git. A fresh checkout needs the baseline artifact placed at that path before baseline viewer or diagnostic commands can load it.

## Runtime Defaults

- `Go1Env()` constructs the default environment directly.
- The default actuator is `actuator_net`.
- The only alternate actuator is `torque_limited_pd`.
- Contact is flat rigid ground with fixed friction `0.8`.
- The viewer and diagnostic scripts no longer require stack-selection flags.
- The default command sampler includes zero, sagittal, lateral, yaw, and mixed commands.
- SCM is the active deformable-terrain backend and is opt-in with `--env-backend scm`.
- CRM/SPH was tested but is too computationally heavy for practical fine-tuning on this setup.

## Core Timing

```text
policy_dt:  0.02 s
policy Hz:  50
physics_dt: 0.005 s
physics Hz: 200
substeps:   4 per policy action
episode:    1000 policy steps
```

The policy emits action targets at 50 Hz. The actuator and Chrono integration run through the four 200 Hz physics substeps.

## Actuator

`actuator_net` uses:

- `resources/actuator_nets/unitree_go1.pt`
- position error history: `q - q_target`
- velocity history
- history update once per 50 Hz policy step
- torque application through Chrono force motors
- URDF effort clipping

`torque_limited_pd` remains for controlled comparisons.

## Reproduction Commands

Static check:

```bash
python -m py_compile go1_env.py go1_scm_env.py train_stand.py diagnose_policy.py view_stand_policy.py view_scm_policy_vsg.py diagnostics.py ppo_compat.py project_config.py
```

Viewer:

```bash
python view_stand_policy.py
```

Fixed forward diagnostic:

```bash
python diagnose_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0 \
  --episodes 1 \
  --max-steps 1000 \
  --out diagnostics/default_forward_eval \
  --log-every-step
```

SCM diagnostic:

```bash
python diagnose_policy.py \
  --env-backend scm \
  runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0 \
  --episodes 1 \
  --max-steps 1000 \
  --out diagnostics/default_scm_forward_eval \
  --log-every-step
```

## Cleanup State

The repo is now intended to expose only the default stack. Old stack selectors, old reward branches, old terrain paths, and old parser compatibility flags should not be reintroduced unless a new experiment requires a clearly isolated branch.

The main project documentation is [docs/documentation.md](docs/documentation.md). Old standing-only notes are archived separately and are not active guidance.
