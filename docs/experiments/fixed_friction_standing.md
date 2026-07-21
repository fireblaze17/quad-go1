# Fixed-Friction Standing ADR Log

This file records the standing cleanup lineage that produced the last accepted
v2 baseline, the failed first v3 relative-state attempt, and the accepted v3.1
fixed-standing recovery. Each entry is an
architecture/experiment decision record.

## ADR-001: Survival-Only Fixed And AB Baselines Were Not Clean Standing

**Status:** Historical / Archived

**Context:** Early fixed-friction and AB friction policies survived evaluation
episodes and looked plausible in the viewer, but later settled-window
diagnostics exposed drift, contact switches, and foot slip.

**Decision:** Treat old fixed and AB checkpoints as historical references, not
active standing baselines.

**Evidence:** Old fixed models reached survival but showed settled displacement,
contact switches, low minimum foot loads, and biased support. AB stayed useful
as pre-filter friction history, but it is not the current source of truth.

**Consequences:** Acceptance now requires settled-window drift/slip/contact
metrics, not only survival.

## ADR-002: Anchor5 Improved Support But Failed Long-Hold Creep

**Status:** Useful intermediate / Rejected as final

**Context:** The planted-foot anchor penalty reduced foot-unload and contact
switch problems. The best support checkpoint was:

```text
runs/stand_fixed_clean_contact2_anchor5_from25k_10k/checkpoints/stand_policy_5000_steps.zip
```

**Decision:** Keep anchor diagnostics and foot-anchor reward, but do not accept
anchor5 as final fixed-friction standing.

**Evidence:** Short behavior was good, but 5000-step long-hold diagnosis showed
slow creep and large accumulated contact-conditioned slip.

**Consequences:** Subsequent work focused on diagnosing the cause of long-hold
creep rather than simply increasing anchor strength.

## ADR-003: Base Drift Weight `10.0` Was Rejected

**Status:** Rejected

**Context:** Increasing `_BASE_DRIFT_PENALTY_WEIGHT` from `2.0` to `10.0` was a
direct attempt to stop long-hold drift.

**Decision:** Do not repeat base drift `10.0` unchanged.

**Evidence:** The stronger base drift penalty worsened drift, contact switching,
and foot unloading.

**Consequences:** Base drift stays at `2.0` with a `0.01 m` deadband. Stronger
stationarity pressure must be tested carefully and one change at a time.

## ADR-004: Freeze-Action Diagnostic Identified Action Jitter

**Status:** Accepted diagnostic

**Context:** The policy was run normally for 1000 steps, then the action was
frozen to the mean action from the previous 100 steps.

**Decision:** Use freeze-action diagnosis to distinguish policy jitter from
contact/pose equilibrium failure.

**Evidence:** Freezing actions reduced settled foot slip from roughly `0.601 m`
to about `0.0047 m`, with nominal classification.

**Consequences:** The main failure was reinterpreted as ongoing action jitter
reaching planted contacts, not an inability to hold a static stance.

## ADR-005: Jitter-Suppression Fine-Tune Became The Intermediate Baseline

**Status:** Accepted intermediate

**Context:** Reward pressure was increased only on action rate and joint
velocity:

```text
_ACTION_RATE_PENALTY_WEIGHT = 0.05
_JOINT_VEL_PENALTY_WEIGHT = 0.02
```

**Decision:** Promote the 5k jitter-suppression checkpoint as the working
baseline before later action-filter work:

```text
runs/stand_jitter_suppression_from_anchor5_10k/checkpoints/stand_policy_5000_steps.zip
```

**Evidence:** It reduced active-reference drift to about `0.91 cm`, kept zero
contact switches, and improved long-hold behavior, but still showed
`one_foot_creep` and about `0.601 m` settled slip.

**Consequences:** Further reward experiments started from this checkpoint, not
from anchor5 or AB.

## ADR-006: Normalized Foot Slip `0.05` Was Rejected

**Status:** Rejected

**Context:** A normalized load-weighted foot-slip penalty was tested to reduce
the remaining one-foot creep.

**Decision:** Do not keep normalized foot-slip weight `0.05` in the active
reward.

**Evidence:** It did not improve the accepted failure mode and later checkpoints
regressed drift/slip behavior.

**Consequences:** `_FOOT_SLIP_PENALTY_WEIGHT` remained `0.00` for the accepted
v2 baseline; foot slip was treated primarily as a diagnostic rather than an
active reward in that stage.

## ADR-007: Stance-Shape `0.05` And `0.005` Were Rejected

**Status:** Rejected

**Context:** A relative-foot stance-shape penalty was tested to reduce the
dominant-foot creep without using global foot-slip pressure.

**Decision:** Remove stance-shape constants, metadata, reward terms, and
diagnostics from active code.

**Evidence:** Both `0.05` and `0.005` failed to beat the jitter baseline.

**Consequences:** Stance-shape is not active and should not be reintroduced
unless a new, specific hypothesis justifies it.

## ADR-008: Eval-Only Action Filter Sweep Was Accepted

**Status:** Accepted

**Context:** The freeze-action diagnostic implied that high-frequency policy
updates caused planted-foot microslip. An eval-only low-pass filter was swept:

```text
tau = 0.02, 0.05, 0.08, 0.12, 0.20 seconds
```

**Decision:** Use `tau=0.05` as the smallest clean action-filter value.

**Evidence:**

```text
tau=0.02: rejected; drift/slip/contact switches worsened
tau=0.05: nominal, slip 0.003434 m, switches 0, active-ref drift 0.001549 m
tau=0.08/0.12/0.20: also nominal, but slower than needed
```

**Consequences:** `action_filter_tau=0.05` became part of the standing control
interface. The policy should be evaluated, viewed, and trained with this filter
unless an explicit ablation disables it.

## ADR-009: Filtered Fine-Tune Promoted The 2k Checkpoint

**Status:** Accepted fixed-friction fallback

**Context:** The action filter was moved into `Go1Env` as an opt-in environment
feature and a short conservative fine-tune was run from the jitter baseline:

```text
runs/stand_action_filter_tau005_from_jitter5k_5k
```

**Decision:** Promote the 2k checkpoint:

```text
runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip
```

**Evidence:** Thirty fixed-0.8, 5000-step episodes with `--action-filter-tau
0.05` produced:

```text
failure_type_counts: {'nominal': 30}
survival_rate: 1.000
active-reference drift: 0.001637 m
settled total contact foot slip: 0.003516 m
settled contact switches: 0
settled min foot load: 26.68 N
```

**Consequences:** Fixed-friction `mu=0.8` standing is accepted. This checkpoint
became the clean fixed fallback and source for randomized-friction training.

## ADR-010: Deferred Jitter Rewards And Friction-Usage Penalties

**Status:** Not tried / Deferred

**Context:** The pasted plan proposed action-anchor, action-acceleration,
max-leg action-rate, and friction-usage penalties if the action filter was not
enough.

**Decision:** Do not add those terms now.

**Evidence:** The action filter alone solved the fixed-0.8 long-hold failure
cleanly.

**Consequences:** Keep the reward simpler. Revisit these terms only if friction
randomization reintroduces action-jitter or contact-usage failures.

## ADR-011: Fixed-Friction Bridge Before Randomized Friction

**Status:** Historical, superseded by ADR-012

**Context:** The fixed `mu=0.8` standing baseline is now clean with
`action_filter_tau=0.05`, but randomized-friction training should not be used
to hide an unresolved fixed-friction problem. Chrono's SMC contact material
composition uses `ChContactMaterialCompositionStrategy::CombineFriction`, which
combines two contacting material frictions with `min(a, b)`.

The active model uses:

```text
ground friction: sampled from friction_range
foot friction: 0.9
effective friction: min(ground friction, 0.9)
contact method: SMC
SetFriction note: current setup uses the same Chrono friction value for static/sliding contact
```

Foot friction `0.9` is a dry rubber-foot assumption for this phase, not a
measured Unitree Go1 value.

**Decision:** Freeze the accepted filtered 2k checkpoint and run fixed
effective-friction bridge slices before PPO updates:

```text
policy: runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip
action_filter_tau: 0.05
main slices: ground mu = 0.6, 0.7, 0.8, 0.9
episodes: 30 deterministic episodes per slice
max_steps: 5000
```

Because foot friction caps the effective value at `0.9`, `ground mu=1.0` and
`1.1` are optional saturation checks, not the main training range. A lower
`mu=0.5` run is an optional stress check.

If `0.6-0.9` passes, start conservative randomized-friction fine-tuning over
`0.6-0.9`. If only `0.7-0.9` is clean, start with `0.7-0.9` and diagnose the
low-friction slice before widening.

**Evidence:** Local PyChrono/header inspection showed the default material
composition strategy returns the minimum friction. The accepted filtered
baseline already passed fixed `mu=0.8`; the open question is whether it remains
clean across nearby fixed effective frictions.

**Consequences:** Diagnostics now log configured ground friction, configured
foot friction, effective friction, contact material metadata, and per-foot
friction usage:

```text
friction_usage = tangential_contact_force / (effective_mu * normal_force + eps)
```

Randomized checkpoints should be promoted by worst fixed-slice behavior, not by
average randomized reward. Stop or reject a run if nominal `mu=0.8` regresses,
contact switches return, loaded-foot slip grows, or friction usage repeatedly
approaches/exceeds `1.0`.

This plan produced the `0.6-0.9` randomized checkpoint documented in ADR-012,
then foot friction was raised to `2.0` so the target range could extend above
effective `0.9`.

## ADR-012: Randomized Friction 0.6-0.9 Promoted The 10k Checkpoint

**Status:** Accepted

**Context:** The filtered fixed checkpoint solved nominal `mu=0.8`, but the
policy needed robustness across a meaningful flat-ground friction range.
Training too long had already caused regressions in earlier experiments, so
checkpoint selection had to use fixed-slice diagnostics rather than final PPO
step count or average reward alone.

**Decision:** Fine-tune from the filtered fixed 2k checkpoint with conservative
PPO over ground friction `0.6-0.9`:

```text
load: runs/stand_action_filter_tau005_from_jitter5k_5k/checkpoints/stand_policy_2000_steps.zip
save_dir: runs/stand_friction_random_060_090_tau005_from_filtered2k
timesteps: 50000
checkpoint_freq: 5000
learning_rate: 0.00005
clip_range: 0.05
target_kl: 0.01
action_filter_tau: 0.05
```

Promote the 10k checkpoint:

```text
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
```

**Evidence:** The 10k checkpoint preserved the clean fixed-slice behavior while
later checkpoints did not improve the worst-slice margin enough to justify
promotion. It also passed probes below the original training range, including
`mu=0.5` and `0.55`, before the foot-friction cap was removed.

**Consequences:** The accepted v2 friction-trained source is the 10k checkpoint,
not the final 50k model. Later checkpoints were not assumed better just because
they had more PPO updates. This remains the accepted friction checkpoint for the
v2/pre-relative-state lineage, while active v3.1 must be evaluated separately
because its observation and reward design changed.

## ADR-013: Foot Friction 2.0 Removes The Effective-Friction Cap

**Status:** Accepted experiment setting

**Context:** The `0.6-0.9` randomized checkpoint promoted at 10k steps passed
fixed-slice probes at `mu=0.5` and `0.55`, and all main slices from `0.6-0.9`.
The next target is effective friction `0.5-1.2`. Chrono combines material
friction with `min(ground, foot)`, so the old foot friction `0.9` capped all
ground values above `0.9`.

**Decision:** Set hardcoded foot friction to `2.0` instead of adding a CLI flag.
This is a cap-removal modeling setting, not a measured Unitree Go1 material.
For all target ground values `0.5-1.2`, effective friction is now the ground
friction.

**Evidence:** Local PyChrono/header checks showed `CombineFriction(a, b)` uses
`std::min`. With foot friction `2.0`, each target slice in `0.5-1.2` is no
longer capped by the foot material.

**Consequences:** Evaluate the v2 randomized 10k checkpoint across
`0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2` before training. If any slice fails, train
from the clean filtered fixed baseline with lower LR, not from the already
randomized `0.6-0.9` checkpoint.

The v2 evaluation recorded nominal outcomes on all slices, so the fallback
training branch was not run:

```text
slices: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
episodes: 30 per slice
failure_type_counts: {'nominal': 30} on every slice
survival_rate: 1.000 on every slice
worst active-reference drift: 0.001558 m
worst settled total contact foot slip: 0.003871 m
settled contact switches: 0 on every slice
worst settled min foot load: 28.53 N
max settled friction usage: 0.01833
```

The accepted v2 checkpoint for this branch was:

```text
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
action_filter_tau = 0.05
_FOOT_FRICTION = 2.0
```

## ADR-014: V2 Reset-State Noise Accepted Through RN-2

**Status:** Accepted for v2 lineage

**Context:** In the v2 branch, reset-state noise was evaluated after the
friction-range work. Reset noise tests whether the controller has a local
recovery basin around the standing pose.

External reset-noise guidance recommended component ablations and stronger
episode counts than the friction gate.

**Decision:** Add opt-in reset-noise levels to the environment and CLI tools,
defaulting to clean behavior, then evaluate before training:

```text
--reset-noise-level clean|rn1|rn2|rn3
--reset-noise-components combined|joint_pos|joint_vel|roll_pitch|base_height|base_velocity
```

Global X/Z translation and yaw are intentionally off for the first reset-noise
pass. On flat standing, they add little physical value and can accidentally
turn a standing recovery test into a world-position tracking or slip objective.

The default ranges are:

```text
RN-1: height +/-0.005 m, roll/pitch +/-1 deg, joint pos +/-0.02 rad,
      joint vel +/-0.05 rad/s, base x/z vel +/-0.02 m/s,
      base y vel +/-0.005 m/s, base angular x/z vel +/-0.05 rad/s

RN-2: height +/-0.010 m, roll/pitch +/-2 deg, joint pos +/-0.04 rad,
      joint vel +/-0.15 rad/s, base x/z vel +/-0.05 m/s,
      base y vel +/-0.02 m/s, base angular x/z vel +/-0.10 rad/s

RN-3: height +/-0.015 m, roll/pitch +/-4 deg, joint pos +/-0.08 rad,
      joint vel +/-0.30 rad/s, base x/z vel +/-0.10 m/s,
      base y vel +/-0.05 m/s, base angular x/z vel +/-0.20 rad/s
```

RN-3 was implemented as an optional stretch test. It was not part of the v2
acceptance gate.

**Evidence:** The implementation evidence:

The current Chrono reset path rebuilds the full system and uses
`DoAssembly` to place the robot in the home pose. Reset joint-position noise is
therefore implemented by assembling the robot into noisy motor targets, then
restoring motor targets to the nominal home pose before the first control step.
Base velocity noise uses Chrono body velocity setters. Joint-velocity noise is
implemented as small child-link angular velocity perturbations because the
imported generic `ChLinkMotor` wrapper does not expose a direct joint-velocity
reset setter.

Evaluation evidence:

```text
policy: runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
action_filter_tau: 0.05
friction slices: 0.5, 0.8, 1.2
reset levels: clean, RN-1, RN-2
keeper episodes: 100 per condition
failure_type_counts: {'nominal': 100} on every condition
survival_rate: 1.000 on every condition
worst active-reference drift: 0.002813 m
worst settled total contact foot slip: 0.003756 m
settled contact switches: 0 on every condition
worst settled min foot load: 28.53 N
max settled friction usage: 0.02035
max non-foot load: 0.0
```

Strongest single evidence file:

```text
diagnostics/keeper_reset_rn2_mu_0p5_confirm100/summary.json
```

Component ablations at fixed `mu=0.8` also passed for joint position, joint
velocity, roll/pitch, base height, and base velocity perturbations.

**Consequences:** The v2 randomized 10k checkpoint is accepted as reset-noise
capable through RN-2 in the pre-v3 code state. This does not transfer to active
v3.1 because v3.1 changed observation shape and reward design. The current v3.1
roadmap must define and test RN1/RN2 directly before moving to push recovery,
friction randomization, or observation noise.

## ADR-015: Relative-State Standing V3 Attempt Stopped At Fixed-Mu Gate

**Status:** Rejected current recipe / keep implementation

**Context:** The accepted v2 standing checkpoint works well, but it was trained
with absolute world base position in the policy observation and absolute world
height in termination. That is not a good foundation for observation noise,
height-shifted terrain, SCM/deformable terrain, or locomotion. A new v3 lineage
was started with this mandatory design rule:

```text
no absolute world X/Y/Z in policy input
no absolute world-height termination
```

**Decision:** Change the environment interface to a 35D relative-state
observation and train v3 from scratch at fixed flat `mu=0.8` before attempting
later robustness gates.

The v3 observation is:

```text
base height relative to support/ground
trunk quaternion
base linear velocity
base angular velocity
12 joint positions
12 joint velocities
```

World X/Z remain available only for diagnostics, reset-reference drift, foot
anchors, and logging. Termination now uses relative trunk height:

```text
relative_height = trunk_world_y - ground_top_y
terminate if relative_height < 0.22
```

`--ground-height-offset`, `--spawn-x`, and `--spawn-z` are evaluation controls.
Defaults preserve the old physical flat-ground setup.

**Evidence:** Implementation validation passed:

```text
observation_space shape: 35
default reset relative height: about 0.34 m
ground-height offset changes world Y but not relative height
spawn offset changes world X/Z without adding those coordinates to observation
static checks passed
```

The first scratch training attempt used:

```bash
python train_stand.py --save-dir runs/stand_v3_relative_obs_fixed08_500k --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 5000 --timesteps 500000 --checkpoint-freq 25000 --learning-rate 0.0003 --clip-range 0.2 --action-filter-tau 0.05
```

Training was stopped at about `198k` timesteps because live rollout length
collapsed after roughly `180k` timesteps. Saved checkpoints through `175k` were
screened with 30 deterministic episodes at fixed `mu=0.8`, clean reset, and
`action_filter_tau=0.05`.

```text
checkpoint   survival   active drift   settled slip   switches   min load   classification
25k          30/30      0.009009 m     0.510896 m     0          21.34 N    nominal, but slip gate failed
50k          30/30      0.114525 m     0.836730 m     0          13.22 N    foot_slip
75k          30/30      0.058233 m     0.759295 m     0          18.13 N    foot_slip
100k         30/30      0.039139 m     0.770571 m     0          20.96 N    foot_slip
125k         30/30      0.066940 m     0.766032 m     0          17.96 N    foot_slip
150k         30/30      0.043649 m     0.545584 m     0          13.47 N    foot_slip
175k         0/30       0.138502 m     0.470033 m     11         0.00 N     foot_slip/fall
```

Diagnostic evidence lives in:

```text
diagnostics/v3_fixed08_025k_clean_mu08_screen30/summary.json
diagnostics/v3_fixed08_050k_clean_mu08_screen30/summary.json
diagnostics/v3_fixed08_075k_clean_mu08_screen30/summary.json
diagnostics/v3_fixed08_100k_clean_mu08_screen30/summary.json
diagnostics/v3_fixed08_125k_clean_mu08_screen30/summary.json
diagnostics/v3_fixed08_150k_clean_mu08_screen30/summary.json
diagnostics/v3_fixed08_175k_clean_mu08_screen30/summary.json
```

**Consequences:** Do not promote a v3 checkpoint from this run. Do not proceed
to later robustness gates from this lineage yet. The relative observation and
relative-height termination code are still the right direction, but the first
scratch recipe is not enough to recover the clean v2 stance.

The likely next v3 attempt should keep the relative-state interface and adjust
the fixed-mu training recipe before another long run. Candidate changes include
starting with the already accepted low-jitter reward/control settings, adding a
small explicit loaded-foot stationarity term during v3 scratch training, or
using a shorter checkpoint-driven curriculum with stricter early evaluation.

## ADR-016: V3.1 Filter-State And Slip-Aligned Reward

**Status:** Accepted for clean fixed `mu=0.8` standing

**Context:** ADR-015 showed that removing absolute world XYZ was the right
research direction, but the first 35D v3 observation/reward recipe let the robot
survive while sliding. The policy no longer saw absolute position, but it also
did not see enough relative support information to align the reward with the
actual settled-slip gate. It also used the action filter as a hidden control
state: the policy output was filtered before reaching the motors, but the policy
did not observe the previous executed action.

The next attempt had to keep the hard v3 rule:

```text
no absolute world X/Y/Z in policy input
no absolute world-height termination
```

**Decision:** Move from the failed 35D v3 observation to a 65D v3.1 observation:

```text
35D relative-state observation
+ 12 previous executed action
+ 2 base X/Z error relative to active standing reference, support frame, /0.03
+ 8 per-foot anchor X/Z errors, support frame, /0.03
+ 4 normalized foot loads: clip(load / 20N, 0, 3)
+ 4 contact flags: load >= 20N
```

World X/Z remain excluded from the policy. They are still used for diagnostics,
reset-reference drift, foot anchors, and logging. Base height remains relative
to the ground top, and termination uses relative height.

The reward was changed to match the acceptance gates more directly:

```text
loaded-foot slip:
  loaded = foot_load >= 20N
  step_slip = sum(loaded * max(norm(foot_xz_now - foot_xz_prev) - 1e-5, 0))
  penalty = 50.0 * (step_slip / 0.03)

foot anchor:
  0.10 * mean(max(0, foot_anchor_displacement / 0.005 - 1)^2)

base drift:
  0.05 * max(0, active_ref_drift / 0.01 - 1)^2

foot load:
  missing = max(0, 25N - foot_load) / 25N
  mean(missing^2) + 2.0 * max(missing^2)

contact/anchor events:
  0.10 per hysteresis contact switch
  0.50 per anchor reset
  1.00 per anchor deactivation

filter-aware action terms:
  0.02 * mean((raw_action - previous_raw_action)^2)
  0.02 * mean((raw_action - executed_action)^2)
```

The hard post-step-100 standing-quality switch was replaced with ramps:

```text
load quality ramp:   clip(step / 50, 0, 1)
stance quality ramp: clip(step / 100, 0, 1)
```

Training used a short fixed-friction run from scratch:

```bash
python train_stand.py --save-dir runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k --terrain flat --friction-min 0.8 --friction-max 0.8 --max-steps 5000 --timesteps 50000 --checkpoint-freq 5000 --learning-rate 0.0003 --clip-range 0.2 --action-filter-tau 0.05
```

**Evidence:** Static and implementation validation passed:

```text
observation_space shape: 65
default reset relative height: about 0.34 m
previous executed action is zero at reset and updates after the first step
static checks passed
```

Every 5k checkpoint through 50k survived the 10-episode fixed `mu=0.8` screen.
The earliest passing checkpoint was promoted for the 30-episode confirmation:

```text
checkpoint: runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip
episodes: 30
result: 30/30 nominal
active-reference drift: 0.000207 m
settled total contact foot slip: 0.007678 m
settled contact switches: 0
settled min foot load: 26.95 N
max settled friction usage: 0.03711
max non-foot load: 0.0
mean filter lag: 0.000110
max filter lag: 0.001470
```

Primary evidence files:

```text
diagnostics/v3p1_fixed08_005k_clean_mu08_confirm30/summary.json
```

**Consequences:** Promote the v3.1 5k checkpoint as the active clean-standing
relative-state baseline for the Chrono position-motor runtime. Do not
treat it as friction-randomized or reset-noise accepted. ADR-018 later corrected
the friction-slice interpretation: friction is not accepted until pushes or
other horizontal disturbances make it meaningful.

This also confirms that v3 did not need a full 500k run to show signal once the
observation and reward matched the slip/drift gates. Future v3.1 continuations
should stay checkpoint-driven and promote the earliest checkpoint that passes
the hard metrics, not the highest reward checkpoint.

## ADR-017: V3.1 Fixed-Friction Slice Interpretation

**Status:** Retracted by ADR-018

**Context:** V3.1 fixed `mu=0.8` clean standing passed in ADR-016. The next
planned gate was to adapt the old v2 friction workflow to the active 65D v3.1
policy. Because v2 history showed that longer PPO training can degrade stance
quality, the v3.1 friction plan was evaluation-first: train only if fixed
slices failed.

The target effective range stayed the same as v2:

```text
ground/effective friction mu = 0.5-1.2
foot friction = 2.0
action_filter_tau = 0.05
reset = clean
max_steps = 5000
```

Chrono combines flat-ground and foot material friction with the minimum of the
two contacting materials in this SMC setup. Setting foot friction to `2.0` is a
cap-removal modeling choice: every commanded ground slice from `0.5` through
`1.2` becomes the effective friction under test instead of being clipped by the
foot material.

**Decision:** Keep the v3.1 5k checkpoint and evaluate it directly over fixed
friction slices before training. Also remove the remaining hard settled-quality
early return from the reward path. The reward now uses the existing ramps
continuously:

```text
load quality ramp:   clip(step / 50, 0, 1)
stance quality ramp: clip(step / 100, 0, 1)
```

Reference-dependent anchor/base penalties remain zero until the standing
reference is captured around step 100. Direct loaded-foot slip and contact
switch penalties ramp smoothly from the beginning.

**Evidence:** Static checks and implementation validation passed:

```text
observation_space shape: 65
reference-dependent penalties finite before and after capture
load_quality_scale and stance_quality_scale present in reward_terms
no hard settled-quality early return remains
```

The 30-episode bridge screen recorded nominal outcomes on every slice:

```text
mu: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
episodes: 30 per slice
failure_type_counts: {'nominal': 30} on every slice
survival_rate: 1.000 on every slice
worst active-reference drift: 0.000215 m
worst settled total contact foot slip: 0.007615 m
settled contact switches: 0 on every slice
worst settled min foot load: 26.95 N
max settled friction usage: 0.03424
```

The 100-episode keeper confirmation also recorded nominal outcomes on every slice:

```text
mu: 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2
episodes: 100 per slice
failure_type_counts: {'nominal': 100} on every slice
survival_rate: 1.000 on every slice
worst active-reference drift: 0.000215 m
worst settled total contact foot slip: 0.007615 m
settled contact switches: 0 on every slice
worst settled min foot load: 26.95 N
max settled friction usage: 0.03424
max non-foot load: 0.0
```

Per-slice keeper evidence:

```text
mu   eff_mu_min/max  episodes  nominal  mean_reward  drift_m   slip_m    switches  min_load_N  max_friction_usage  worst_foot
0.5  0.5 / 0.5       100       100/100   5679.921     0.000201  0.007565  0         26.946      0.034239            RR
0.6  0.6 / 0.6       100       100/100   5680.691     0.000200  0.007615  0         26.951      0.026631            RR
0.8  0.8 / 0.8       100       100/100   5679.741     0.000215  0.007392  0         26.977      0.025404            RR
0.9  0.9 / 0.9       100       100/100   5679.741     0.000215  0.007392  0         26.977      0.022581            RR
1.0  1.0 / 1.0       100       100/100   5679.741     0.000215  0.007392  0         26.977      0.020323            RR
1.1  1.1 / 1.1       100       100/100   5679.741     0.000215  0.007392  0         26.977      0.018476            RR
1.2  1.2 / 1.2       100       100/100   5679.741     0.000215  0.007392  0         26.977      0.016936            RR
```

Friction margin and action smoothness:

```text
mu   mean_friction_usage  frames>0.7  frames>0.9  frames>1.0  raw_dact_mean  raw_dact_max  exec_dact_mean  exec_dact_max  filter_lag_mean  filter_lag_max
0.5  0.009106             0           0           0           0.000186        0.001705      0.000004        0.000039       0.000112         0.000963
0.6  0.007564             0           0           0           0.000174        0.001426      0.000004        0.000029       0.000108         0.000729
0.8  0.005511             0           0           0           0.000170        0.001907      0.000004        0.000046       0.000104         0.001150
0.9  0.004898             0           0           0           0.000170        0.001907      0.000004        0.000046       0.000104         0.001150
1.0  0.004409             0           0           0           0.000170        0.001907      0.000004        0.000046       0.000104         0.001150
1.1  0.004008             0           0           0           0.000170        0.001907      0.000004        0.000046       0.000104         0.001150
1.2  0.003674             0           0           0           0.000170        0.001907      0.000004        0.000046       0.000104         0.001150
```

Field definitions:

```text
drift_m = settled base displacement from active standing reference
slip_m = settled total loaded-contact foot slip distance
switches = settled contact switches
min_load_N = settled minimum foot load
max_friction_usage = max settled ||Ft|| / (mu * Fz + eps)
raw_dact = raw policy action delta
exec_dact = filtered/executed action delta
filter_lag = raw action minus executed action
```

Primary evidence files:

```text
diagnostics/v3p1_friction_keeper_mu_0p5_confirm100/summary.json
diagnostics/v3p1_friction_keeper_mu_0p6_confirm100/summary.json
diagnostics/v3p1_friction_keeper_mu_0p8_confirm100/summary.json
diagnostics/v3p1_friction_keeper_mu_0p9_confirm100/summary.json
diagnostics/v3p1_friction_keeper_mu_1p0_confirm100/summary.json
diagnostics/v3p1_friction_keeper_mu_1p1_confirm100/summary.json
diagnostics/v3p1_friction_keeper_mu_1p2_confirm100/summary.json
```

**Consequences:** This decision was later retracted. The diagnostics remain
historical data, but the interpretation was wrong: quiet fixed-friction slices
do not prove friction robustness when the standing task creates little or no
meaningful horizontal shear. See ADR-018.

## ADR-018: Friction Slice Claim Retracted Until Pushes Make Friction Meaningful

**Status:** Accepted correction

**Context:** V3.1 clean standing at fixed `mu=0.8` is useful because it fixed
the original clean-standing problems: action-jitter-driven creep, contact
switching, foot unloading, base drift, loaded-foot slip, and non-foot support.
After that, fixed-friction slices from `mu=0.5` to `1.2` were run and initially
described as friction robustness.

That interpretation was mistaken. In quiet standing, the robot is not being
pushed and is not required to create meaningful horizontal shear. A
snap-to-home zero-action controller can also stand in similar quiet conditions.
Under those conditions, changing friction values does not demonstrate that the
learned policy can recover from slip, reject disturbances, or use friction
robustly.

**Decision:** Retract the V3.1 friction-robustness claim. The accepted V3.1
claim is clean standing only under the Chrono position-motor actuator:

```text
fixed mu=0.8 baseline condition
clean reset
no pushes
no reset-noise acceptance
no friction-randomization acceptance
no absolute world XYZ in policy input
```

Friction randomization and fixed-friction slice tests should not be used as
standing-policy acceptance gates until random pushes or another disturbance
create horizontal shear demands. Friction becomes meaningful after the policy is
asked to recover from horizontal disturbance; before that, it is mostly a
non-stressful parameter for this task.

**Evidence:** The clean-standing diagnostic remains the accepted V3.1 evidence:

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

These metrics directly address the clean-standing failures that drove the v3.1
work. The fixed-friction slice numbers from ADR-017 remain historical logs, but
they are not pass evidence for friction robustness.

**Consequences:** Documentation and roadmap language must describe V3.1
friction as unaccepted. The current order is:

```text
1. define, test, and possibly train RN1/RN2 reset recovery
2. random push recovery
3. friction randomization after pushes make friction meaningful
4. observation noise
```

Do not add reset-noise acceptance, push acceptance, or friction-randomization
acceptance until those conditions are directly tested.

## ADR-019: Reset-Noise Ranges Updated For Standing-Only Recovery

**Status:** Implemented, not accepted

**Context:** The old V3.1 RN1/RN2 reset-noise ranges were very small, omitted
base yaw, and did not perturb base X/Z position. External standing-policy advice
recommended using full yaw immediately because yaw is rotation about gravity,
while keeping roll/pitch bounded so the task remains standing recovery rather
than fall recovery. It also recommended additive joint-position noise rather
than multiplicative scaling around the default pose.

**Decision:** Redefine RN1/RN2 for standing-only reset recovery:

```text
RN1:
  base X/Z position:        +/-0.03 m
  base height Y:            +/-0.015 m
  yaw about gravity:        [-pi, pi]
  roll/pitch:               +/-0.05 rad
  base linear X/Z velocity: +/-0.10 m/s
  base linear Y velocity:   +/-0.03 m/s
  base angular X/Z:         +/-0.15 rad/s
  base angular Y/yaw:       +/-0.20 rad/s
  joint pos hip/thigh/knee: +/-0.04 / +/-0.08 / +/-0.10 rad
  joint velocity:           +/-0.20 rad/s

RN2:
  base X/Z position:        +/-0.10 m
  base height Y:            +/-0.030 m
  yaw about gravity:        [-pi, pi]
  roll/pitch:               +/-0.12 rad
  base linear X/Z velocity: +/-0.25 m/s
  base linear Y velocity:   +/-0.05 m/s
  base angular X/Z:         +/-0.40 rad/s
  base angular Y/yaw:       +/-0.50 rad/s
  joint pos hip/thigh/knee: +/-0.10 / +/-0.12 / +/-0.15 rad
  joint velocity:           +/-0.50 rad/s
```

Chrono is Y-up, so base height is Chrono `Y`, horizontal position is Chrono
`X/Z`, and yaw is rotation about Chrono `Y`. The CLI component selector was
expanded to:

```text
combined|joint_pos|joint_vel|roll_pitch|yaw|base_height|base_position|base_velocity
```

**Evidence:** Implementation now logs explicit reset samples for base X/Z
offset, height offset, roll, pitch, yaw, per-axis base linear velocity, per-axis
base angular velocity, joint-position RMS, and joint-velocity RMS. Smoke
diagnostics should be used to verify the ranges are actually sampled before any
training or acceptance claim:

```bash
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 3 --max-steps 1000 --action-filter-tau 0.05 --reset-noise-level rn1 --reset-noise-components combined --out diagnostics/v3p1_rn1_new_ranges_smoke3
python diagnose_policy.py runs/stand_v3p1_relative_obs65_slip_anchor_fixed08_50k/checkpoints/stand_policy_5000_steps.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 3 --max-steps 1000 --action-filter-tau 0.05 --reset-noise-level rn2 --reset-noise-components combined --out diagnostics/v3p1_rn2_new_ranges_smoke3
```

**Consequences:** RN1/RN2 are now meaningful standing-reset definitions for the
V3.1 branch, but they are not accepted. Reset-noise robustness still requires
direct screens and, if necessary, training. RN3 remains a temporary/debug-only
deterministic upper-RN2 probe.

Follow-up implementation note: reset-noise builds now apply a whole-robot
contact-safety lift after assembly if any foot would start below the configured
clearance. This prevents RN1/RN2 results from being polluted by accidental
foot-ground clipping while preserving the sampled pose geometry.

## ADR-020: Separate Physics Rate From Policy Control Rate

**Status:** Implemented, requires retraining/revalidation

**Context:** Earlier standing runs updated the policy action every physics step.
That made the policy/control interface effectively very high frequency. The
project then spent a lot of reward and control design effort suppressing action
jitter, especially with action-rate penalties and the `tau=0.05` action
low-pass filter.

After comparing against a more standard simulation loop, the better modeling
choice became clear: physics should integrate faster than the policy updates.
The simulator should handle fine contact/rigid-body integration, while the RL
policy should act at a slower controller rate.

**Decision:** Change the active environment timing to:

```text
physics timestep = 0.005 s  (200 Hz)
control timestep = 0.020 s  (50 Hz)
physics substeps per policy action = 4
1000 RL steps = 20 s simulated time
```

The policy now chooses one action per control step. Chrono holds the same
position target while it advances four physics substeps.

**Evidence:** Static checks passed and environment metadata reports:

```text
control_time_step = 0.02
control_frequency = 50.0
physics_time_step = 0.005
physics_frequency = 200.0
physics_substeps = 4
```

A one-episode compatibility smoke with the old V3.1 5k checkpoint survived but
did not preserve the old slip metric:

```text
diagnostics/control50hz_retiming_smoke1/summary.json
episodes: 1
result: nominal
active-reference drift: 0.000059 m
settled total contact foot slip: 0.238558 m
settled contact switches: 0
settled min foot load: 26.87 N
action_filter_tau: 0.05
alpha: 0.285714
```

**Consequences:**

- The 50 Hz timing is a simulator/control-interface correction, not a reward
  trick.
- The older reward function and action-filter settings were partly compensating
  for too-fast policy updates.
- Old V3.1 checkpoints are useful source/history, but they are not automatically
  accepted under the new timing.
- The next training run should simplify or retune reward terms for the slower
  control rate instead of continuing to pile on smoothness rewards.
