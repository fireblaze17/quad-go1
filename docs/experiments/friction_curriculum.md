# Flat Friction Curriculum

## Summary

The current standing v2 checkpoint is the fixed-friction anchor. The flat
friction curriculum continues training from that anchor through three gradually
wider friction ranges. Each stage writes to a new run directory so the accepted
baseline is not overwritten.

```text
runs/stand_base_v2
-> runs/stand_friction_a_07_09
-> runs/stand_friction_ab_065_095  (accepted as B-capable)
-> runs/stand_friction_c_05_11
```

Use `friction_curriculum.py` to prepare the base copy and print or run the exact
commands.

All commands below assume the WSL conda environment is active:

```bash
conda activate chrono-go1
```

Shared baseline paths and run directories are defined in `project_config.py`.
`friction_curriculum.py` uses those constants so friction C loads the accepted
B-capable AB checkpoint by default. Friction B is marked accepted via AB rather
than retrained into the rejected B continuation folders.

## Stage 0: Preserve The Base

```bash
python friction_curriculum.py prepare-base
```

This copies:

```text
runs/stand/final_model.zip
```

to:

```text
runs/stand_base_v2/final_model.zip
```

If `args.json` or `env_constants.json` exist beside the source model, they are
copied too. Otherwise the script writes a short local `baseline_notes.md`.

## Stage 1: Friction A

Range: `0.7-0.9`

Status: accepted from a 300k-step continuation run.

Accepted checkpoint:

```text
runs/stand_friction_a_07_09/final_model.zip
```

Accepted randomized-range evaluation:

```text
survival_rate: 1.000
mean_length: 1000.0
termination_reasons: {'truncated': 10}
min_upright_score: 0.995
foot_contact_error: 0.005120
min_foot_load: 23.836766
friction_min_seen: 0.711
friction_max_seen: 0.890
```

Accepted fixed-0.8 regression:

```text
survival_rate: 1.000
mean_length: 1000.0
termination_reasons: {'truncated': 10}
min_upright_score: 0.995
foot_contact_error: 0.004596
min_foot_load: 23.885856
friction_min_seen: 0.800
friction_max_seen: 0.800
```

What changed:

- Continued PPO training from `runs/stand_base_v2/final_model.zip`.
- Randomized flat-ground friction over `0.7-0.9`.
- Kept reward weights, collision whitelist, home pose, action scale, contact
  settings, and solver settings unchanged.
- Forced Stable-Baselines3 PPO to use CPU for this MLP policy after SB3 warned
  that CUDA can be slower for non-CNN PPO.

Why it changed:

- The fixed-friction standing v2 policy was accepted, but it only covered
  friction `0.8`.
- The next robustness step is friction randomization before changing terrain,
  reset noise, or reward terms.
- Native GPU use was not useful here because Chrono rollout simulation and MLP
  PPO updates are CPU-oriented; the warning coincided with lower training FPS.

What worked:

- A 300k-step continuation was visibly cleaner than a 150k-step continuation.
- The 300k policy kept all four feet loaded, stayed upright, and did not show
  the in-place shuffling/tilt seen in the 150k run.
- Fixed-friction `0.8` behavior did not regress relative to the friction A
  randomized-range evaluation.

What did not work:

- The 150k run survived all episodes, but it was not accepted. It shuffled in
  place and leaned visually.
- Its diagnostics agreed with the viewer: lower minimum upright score, much
  larger foot-contact error, lower minimum foot load, more action-rate motion,
  higher angular velocity, and more X/Z motion.

150k rejected signature:

```text
survival_rate: 1.000
mean_length: 1000.0
min_upright_score: 0.983
foot_contact_error: 0.225816
min_foot_load: 5.210639
mean_abs_action_delta: 0.091662
mean_abs_ang_vel: 0.418254
mean_abs_xz_vel: 0.041915
```

Tradeoffs and consequences:

- 300k is longer than the original fixed-friction run, but that is acceptable:
  fine-tuning means starting from an existing policy, not necessarily using
  fewer timesteps.
- More PPO steps can cause policy drift, so acceptance still depends on
  evaluation, viewer behavior, and contact diagnostics, not reward alone.
- The canonical friction A path now points at the accepted 300k checkpoint:
  `runs/stand_friction_a_07_09/final_model.zip`.
- Older comparison folders may exist locally, but downstream curriculum stages
  should load the canonical accepted path only.

Print command:

```bash
python friction_curriculum.py train friction_a
```

Run command:

```bash
python friction_curriculum.py train friction_a --run
```

Evaluate:

```bash
python friction_curriculum.py eval friction_a --run
```

View:

```bash
python friction_curriculum.py view friction_a --run
```

## Stage 2: Friction B

Range: `0.6-1.0`

Status: accepted via AB checkpoint generalization.

Accepted checkpoint:

```text
runs/stand_friction_ab_065_095/final_model.zip
```

Accepted randomized-range evaluation on `0.6-1.0`:

```text
survival_rate: 1.000
mean_length: 1000.0
termination_reasons: {'truncated': 30}
min_upright_score: 0.999
foot_contact_error: 0.009621
min_foot_load: 23.393599
mean_abs_xz_vel: 0.011832
tilt_error: 0.001289
leg_symmetry_error: 0.000455
friction_min_seen: 0.610
friction_max_seen: 0.999
```

Headless diagnosis on `0.6-1.0`:

```text
survival_rate: 1.000
mean_length: 1000.0
cause_counts: {'no_tilt_threshold_crossing': 30}
max_tilt_error: 0.001971
min_upright_score: 0.999014
max_nonfoot_load: 0.0
```

Viewer check: accepted visually on the full B range with no obvious lean,
vibration, permanent foot unload, or non-foot support.

Why this is accepted:

- AB was trained on `0.65-0.95`, but it generalized cleanly to `0.6-1.0`.
- The full B range is therefore not inherently too hard for the standing policy.
- Additional B fine-tuning made the policy worse, so the accepted checkpoint is
  the clean generalizing checkpoint rather than the most recently trained one.

The direct A-to-B run survived but leaned visibly and showed asymmetric foot
loading. Increasing the tilt penalty to `0.5` made the policy worse, so tilt
stays at the accepted `0.25` value.

Friction A remains the accepted A-stage checkpoint. AB is now the current
default B-capable standing baseline in `project_config.py`, and friction C
loads from AB.

The `sym002` B experiment added a tiny `0.02 * leg_symmetry_error` penalty. It
survived but did not remove the visible lean and produced a worse contact
signature than the clean AB reference, so the active symmetry penalty was
removed again.

B retraining is intentionally disabled in `friction_curriculum.py`. Use the AB
checkpoint for B evaluation/viewing and only create new B experiment folders if
there is a specific research question to test.

Runnable headless diagnosis for the accepted B-capable checkpoint:

```bash
python diagnose_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0 --episodes 30 --out diagnostics/ab_on_b_range
```

The rejected local B/ABB folders were deleted after their failure signatures
were documented. They are intentionally not part of GitHub and should not be
required for reproducing the accepted state.

Initial headless diagnosis:

```text
AB on 0.6-1.0:       no_tilt_threshold_crossing in 30/30 episodes
ABB on 0.625-0.975: foot_unload_before_tilt in 30/30 episodes
B from AB on 0.6-1.0: foot_unload_before_tilt in 30/30 episodes
sym002 on 0.6-1.0:  foot_unload_before_tilt in 30/30 episodes
```

The leaned policies all showed a consistent left-vs-right load bias with FL as
the dominant loaded foot and zero non-foot contact. That points at a learned
support/load-bias attractor during continued training, not hidden calf/thigh/hip
support and not the B friction range being immediately impossible for the AB
policy.

Evaluate the accepted B-capable checkpoint on the randomized range:

```bash
python evaluate_stand.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0 --episodes 30
```

Also sanity-check the original fixed friction:

```bash
python evaluate_stand.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10
```

View with compact and full diagnostics:

```bash
python view_stand_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0
python view_stand_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0 --full-diagnostics
```

## Stage 3: Friction C

Range: `0.5-1.1`

Friction C is the next trainable curriculum stage. It loads from the accepted
AB B-capable checkpoint:

```bash
python friction_curriculum.py train friction_c --run
```

Evaluate:

```bash
python friction_curriculum.py eval friction_c --run
```

Held-out checks:

```bash
python friction_curriculum.py heldout friction_c --run
```

Held-out friction `0.4` and `1.2` are diagnostic only. They help show whether
the learned policy is close to extrapolating beyond the training range.

## Acceptance Criteria

Accept a stage only if:

```text
survival_rate: 1.000
mean_length: 1000.0
termination_reasons: {'truncated': 10}
min_upright_score > 0.99
no visible vibration
no permanent foot unload
no steady foot sliding
nonfoot_load_max = 0
```

Reject or retrain if fixed friction `0.8` becomes worse than standing v2.

## Notes

- Do not change reward weights during this curriculum.
- Do not change collision settings, contact materials, action scale, or home
  pose.
- Use `--load` at every curriculum stage after the base.
- Use a new save directory for each stage.
- Start reset-noise curriculum only after Friction B or C is accepted.
