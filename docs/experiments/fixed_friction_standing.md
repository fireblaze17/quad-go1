# Fixed-Friction Standing ADR Log

This file records the fixed-friction standing cleanup that produced the current
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

**Status:** Accepted current baseline

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

**Consequences:** Fixed-friction `mu=0.8` standing is accepted. The next work is
friction bridge testing, not more fixed-0.8 reward shaping.

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
