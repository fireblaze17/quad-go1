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
-> runs/stand_friction_c_05_11    (trained challenger; currently rejected)
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

Status: AB is the official current baseline on this range. The trained
C-from-AB continuation and three equal-budget scratch-C seeds all survived, but
all were rejected on cleaner-standing criteria.

The accepted AB checkpoint was tested directly on the wider C range before any
new C training:

```bash
python run_regression.py runs/stand_friction_ab_065_095/final_model.zip --name ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100
python view_stand_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.5 --friction-max 1.1
```

AB-on-C evaluation:

```text
episodes: 100
survival_rate: 1.000
mean_reward: 1138.815
mean_length: 1000.0
min_trunk_y: 0.340
min_upright_score: 0.999
foot_contact_error: 0.009350
min_foot_load: 23.431158
mean_abs_xz_vel: 0.011950
max_abs_xz_vel: 0.019967
tilt_error: 0.001282
leg_symmetry_error: 0.000454
termination_reasons: {'truncated': 100}
friction_min_seen: 0.501
friction_max_seen: 1.096
```

AB-on-C diagnosis:

```text
survival_rate: 1.000
mean_length: 1000.0
cause_counts: {'no_tilt_threshold_crossing': 100}
dominant_load_axes: {'diag_vs_diag': 86, 'front_vs_rear': 4, 'left_vs_right': 10}
dominant_loaded_legs: {'FR': 66, 'RL': 31, 'FL': 3}
least_loaded_legs: {'RR': 78, 'FL': 21, 'FR': 1}
worst_case:
  max_tilt_error: 0.001973
  max_load_imbalance: 0.628981
  max_foot_dxz: 0.082529
  max_nonfoot_load: 0.000000
  min_foot_load: 0.000000
  min_upright_score: 0.999013
```

Interpretation:

- AB already satisfies the C-range survival/upright/contact acceptance checks.
- The transient `min_foot_load: 0.0` appears in the worst case, but there is no
  tilt threshold crossing, no non-foot support, and no single FL-heavy support
  attractor like the rejected B continuations.
- Viewer inspection on `0.5-1.1` looked clean, so AB is the active C-capable
  reference until another model beats it.

The first C continuation loaded from the accepted AB B-capable checkpoint:

```bash
python friction_curriculum.py train friction_c --run
```

It saved:

```text
runs/stand_friction_c_05_11/final_model.zip
```

C-from-AB evaluation on `0.5-1.1`:

```text
episodes: 100
survival_rate: 1.000
mean_reward: 1138.796
mean_length: 1000.0
min_trunk_y: 0.337
min_upright_score: 0.999
foot_contact_error: 0.016239
min_foot_load: 20.133857
mean_abs_xz_vel: 0.045557
max_abs_xz_vel: 0.064314
tilt_error: 0.000208
leg_symmetry_error: 0.003477
termination_reasons: {'truncated': 100}
friction_min_seen: 0.506
friction_max_seen: 1.099
```

C-from-AB diagnosis:

```text
survival_rate: 1.000
mean_length: 1000.0
cause_counts: {'no_tilt_threshold_crossing': 100}
dominant_load_axes: {'left_vs_right': 66, 'front_vs_rear': 28, 'diag_vs_diag': 6}
dominant_loaded_legs: {'FL': 85, 'FR': 7, 'RL': 3, 'RR': 5}
least_loaded_legs: {'RR': 79, 'RL': 18, 'FR': 3}
worst_case:
  max_tilt_error: 0.001134
  max_load_imbalance: 0.495874
  max_foot_dxz: 0.416218
  max_nonfoot_load: 0.000000
  min_foot_load: 1.115778
  min_upright_score: 0.999433
```

Why C-from-AB is rejected:

- It survives, but survival is not enough for this standing baseline.
- Mean X/Z velocity is about four times worse than AB-on-C
  (`0.045557` vs `0.011950`), matching the viewer observation that it drifts or
  slides, especially on lower friction samples.
- Foot contact quality regressed (`foot_contact_error` `0.016239` vs
  `0.009350`) and minimum eval foot load dropped (`20.1 N` vs `23.4 N`).
- Diagnosis shows a strong learned left/right support pattern: FL is dominant
  loaded in 85/100 episodes and RR is least loaded in 79/100.
- Worst-case foot X/Z displacement rose to `0.416218`, compared with `0.082529`
  for AB-on-C.

Tradeoff:

- C-from-AB has lower action magnitude and lower tilt error, but it obtains
  that quiet-looking trunk by accepting worse ground-plane drift, foot
  displacement, and asymmetric load. For this project, clean contact and low
  sliding matter more than the small tilt improvement.
- Do not tune reward just to rescue C-from-AB while AB already passes C. Reward
  changes are justified only if AB fails a new requirement or a challenger
  clearly improves an important acceptance metric without introducing worse
  contact behavior.

## Direct Scratch C Comparison

To compare curriculum against direct training fairly, scratch C uses the same
total timestep budget as the accepted staged path through AB:

```text
fixed base 0.8: 100k
friction A:     300k
friction AB:    300k
total:          700k
```

Training commands:

```bash
python train_stand.py --terrain flat --friction-min 0.5 --friction-max 1.1 --save-dir runs/stand_friction_c_scratch_seed1_700k --timesteps 700000 --seed 1 --checkpoint-freq 100000
python train_stand.py --terrain flat --friction-min 0.5 --friction-max 1.1 --save-dir runs/stand_friction_c_scratch_seed2_700k --timesteps 700000 --seed 2 --checkpoint-freq 100000
python train_stand.py --terrain flat --friction-min 0.5 --friction-max 1.1 --save-dir runs/stand_friction_c_scratch_seed3_700k --timesteps 700000 --seed 3 --checkpoint-freq 100000
```

Regression commands:

```bash
python run_regression.py runs/stand_friction_ab_065_095/final_model.zip --name ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_c_05_11/final_model.zip --name c_from_ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_c_scratch_seed1_700k/final_model.zip --name c_scratch_seed1_700k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_c_scratch_seed2_700k/final_model.zip --name c_scratch_seed2_700k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
python run_regression.py runs/stand_friction_c_scratch_seed3_700k/final_model.zip --name c_scratch_seed3_700k_on_c --friction-min 0.5 --friction-max 1.1 --episodes 100
```

`run_regression.py` writes the full stdout from `evaluate_stand.py` and
`diagnose_policy.py`, plus a compact machine-readable summary:

```text
diagnostics/<name>/evaluate_stdout.txt
diagnostics/<name>/diagnose_stdout.txt
diagnostics/<name>/summary.json
diagnostics/<name>/episodes.json
diagnostics/<name>/regression_summary.json
```

Final C-range comparison:

| Model family | Training path | Timesteps | Survival | Upright | Contact error | Min foot load | X/Z drift | Diagnosis pattern | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AB-on-C | fixed 0.8 -> A -> AB, tested on C | 700k staged | 1.000 | 0.999 | 0.009350 | 23.431 N | 0.011950 | no tilt crossing; mostly diagonal load | accepted/current baseline |
| C-from-AB | fixed 0.8 -> A -> AB -> C | 1.0M staged | 1.000 | 0.999 | 0.016239 | 20.134 N | 0.045557 | no tilt crossing, but FL-heavy left/right bias | rejected: drift/sliding and asymmetric support |
| Scratch C seed1 | random init -> C | 700k | 1.000 | 0.996 | 0.016500 | 19.784 N | 0.024861 | foot unload before tilt in 100/100; FL-heavy | rejected: worse contact/upright/load balance |
| Scratch C seed2 | random init -> C | 700k | 1.000 | 0.987 | 0.104047 | 15.162 N | 0.044395 | foot unload before tilt in 100/100; left/right bias | rejected: large contact/upright regression |
| Scratch C seed3 | random init -> C | 700k | 1.000 | 0.974 | 0.249241 | 5.966 N | 0.039411 | foot unload before tilt in 100/100; FL unloaded | rejected: severe contact/upright regression |

Interpretation:

- All three direct scratch-C seeds reached 100/100 survival, so survival alone
  is a weak acceptance criterion for this task.
- All three scratch-C seeds found a repeatable `foot_unload_before_tilt`
  pattern in 100/100 diagnostic episodes. The exact loaded/unloaded legs vary
  by seed, but the family of failure is consistent: asymmetric support first,
  tilt and poorer contact quality after.
- C-from-AB is less obviously tilted than the scratch seeds, but it drifts much
  more than AB and keeps a strong FL-heavy/RR-light pattern.
- AB is accepted because it is the cleanest standing policy under the shared
  C-range battery and viewer check, not because curriculum is assumed to be
  automatically better.
- A future model must beat AB on eval, diagnosis, and viewer behavior before it
  replaces the current baseline. A scratch seed with 100% survival is not enough.

Evaluate C-from-AB:

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

## Current Cleanup Note

- Do not change collision settings, contact materials, action scale, or home
  pose while comparing friction-policy candidates.
- Use `--load` at every curriculum stage after the base.
- Use a new save directory for each stage.
- AB was the correct old winner among the friction A/B/C candidates: C-from-AB
  and scratch-C survived but were rejected on drift, contact, upright, and
  load-bias evidence.
- Later settled-window diagnostics showed that AB itself still creeps/slips
  enough that it should be treated as an archived old baseline, not final-clean
  standing.
- The next friction pass should restart before AB, likely from
  `runs/stand_friction_a_07_09/final_model.zip`, and train a cleaner AB
  replacement before any reset-noise work returns.
- Survival-only is not a valid standing acceptance criterion.
