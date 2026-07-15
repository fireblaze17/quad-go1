# Fixed-Friction Standing ADR Log

This file records the standing cleanup lineage that produced the current
accepted baseline. Each entry is an architecture/experiment decision record.

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

**Consequences:** `_FOOT_SLIP_PENALTY_WEIGHT` remains `0.00`; foot slip remains
a diagnostic rather than an active reward in the current baseline.

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

**Consequences:** The current friction-trained source is the 10k checkpoint, not
the final 50k model. Future randomized runs should be selected by worst fixed
slice, not by longest training or average randomized reward.

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

**Consequences:** Evaluate the current randomized 10k checkpoint across
`0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2` before training. If any slice fails, train
from the clean filtered fixed baseline with lower LR, not from the already
randomized `0.6-0.9` checkpoint.

The evaluation passed all slices, so the fallback training branch was not run:

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

The accepted current baseline is therefore:

```text
runs/stand_friction_random_060_090_tau005_from_filtered2k/checkpoints/stand_policy_10000_steps.zip
action_filter_tau = 0.05
_FOOT_FRICTION = 2.0
```
