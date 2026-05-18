# Training Roadmap

Architecture Decision Records for the Chrono Go1 standing policy. Each section
records the context that forced a decision, what was decided, what worked, what
did not work, and what the decision costs.

Git history records every code state. This file records the current rational
path through the experiments.

## Stage 1: Standing - Flat Terrain, Fixed Friction=0.8

```python
Go1Env(terrain="flat", enable_motors=True, friction_range=(0.8, 0.8))
```

Current reward:

```python
reward = (
    alive_bonus
    + upright_reward
    - pose_penalty
    - control_penalty
    - joint_vel_penalty
    - action_rate_penalty
    - tilt_penalty
    - ang_vel_penalty
    - xz_vel_penalty
    - foot_contact_penalty
)
```

Current weights:

```text
alive_bonus             1.00
upright_reward          0.15 * upright_score
pose_penalty            0.30 * mean(joint_error^2)
control_penalty         0.03 * mean(action^2)
joint_vel_penalty       0.01 * mean(joint_velocity^2)
action_rate_penalty     0.03 * mean(action_delta^2)
tilt_penalty            0.25 * (trunk_x_up^2 + trunk_y_up^2)
angular_vel_penalty     0.01 * mean(trunk_angular_velocity^2)
xz_vel_penalty          0.20 * mean([vx, vz]^2)
foot_contact_penalty    0.10 * mean(missing_foot_load^2)
minimum foot load       20 N per foot before penalty is zero
```

`leg_symmetry_error` remains in reward logs as a diagnostic only. It is not an
active reward term in the accepted baseline.

Current accepted checkpoint:

```text
survival_rate:       1.000
mean_length:         1000.0
min_trunk_y:         0.337
min_upright_score:   1.000
mean_abs_xz_vel:     0.007
mean_abs_joint_vel:  0.393
mean_abs_action:     0.304
mean_foot_load:      32.09 N
min_foot_load:       17.58 N
termination:         truncated only
```

Conclusion: flat-ground standing v2 is accepted. It stands upright, keeps all
four feet near the ground, has no visible vibration in the viewer, and uses zero
non-foot contact after the collision whitelist change.

---

## What Actually Solved The Standing Problem

The final fix was a chain, not a single magic reward weight:

1. Fix the home pose and spawn height so zero action is already physically sane.
2. Fix joint observations so pose error is real, not a frame-sign artifact.
3. Add viewer diagnostics for action rate, joint velocity, foot slip, foot load,
   foot height, and non-foot link contact.
4. Discover that hip/thigh/calf collision contact was supporting the policy.
5. Disable hip/thigh/calf terrain collisions, keeping trunk and feet collidable.
6. Retrain and discover the policy could still unload one foot.
7. Add a weak four-foot support penalty based on contact load.

This is why tuning symmetry, angular velocity, or action rate alone did not
fully solve the issue. Those terms shaped motion, but the root problem was that
the policy had access to bad support modes and did not care whether every foot
was actually participating.

---

## ADR-001: Reward Term - `alive_bonus`

**Status:** Accepted

**Context:** A base-height reward was considered first, but absolute height
assumes a flat Y=0 ground plane. That is fragile for Chrono SCM terrain, where
the foot-ground interface can sink or deform.

**Decision:**

```python
alive_bonus = 1.0
```

**Why it worked:** it creates a simple base objective: remain in a valid standing
episode for 1000 steps.

**Tradeoff:** alive reward alone is sparse. It does not distinguish clean
standing from barely surviving, so shaping terms are required.

---

## ADR-002: Reward Term - `upright_reward`

**Status:** Accepted at low weight

**Context:** With survival alone, the policy can survive while trending toward a
bad posture. The trunk needs a dense verticality signal.

**Decision:**

```python
upright_score = max(0.0, trunk_z_up)
upright_reward = 0.15 * upright_score
```

Chrono is Y-up, so `trunk_z_up` means the trunk local Z axis dotted against
world Y.

**What did not work:** increasing upright weight reduced visible lean in one run
but brought back more leg shuffling. Upright reward is broad: it says "be
vertical" but not "use four clean foot contacts."

**Tradeoff:** the accepted 0.15 weight is intentionally modest. Tilt and support
terms handle more specific failure modes.

---

## ADR-003: Reward Term - `pose_penalty`

**Status:** Accepted at 0.30

**Context:** Early policies could keep the trunk mostly vertical while using odd
leg configurations. Once the joint observation bug was fixed, pose error became
meaningful.

**Decision:**

```python
pose_error = joint_pos - home_joint_angles
pose_penalty = 0.30 * mean(pose_error**2)
```

**What worked:** pose penalty keeps the legs near the known stable Chrono home
stance.

**What did not work:** pose penalty alone did not force the lifted-looking foot
back into useful contact. The policy could still hold a bad support pattern near
home.

**Tradeoff:** pose penalty preserves Go1-like geometry, but too much pose
regularization can fight necessary balance corrections.

---

## ADR-004: Reward Term - `control_penalty`

**Status:** Accepted at 0.03

**Context:** The policy often used large action targets. A control penalty
discourages large sustained commands.

**Decision:**

```python
control_penalty = 0.03 * mean(action**2)
```

**What did not work:** larger control weights reduced corrective authority and
made the robot tip or spin sooner. The problem was not simply "actions are too
large"; the contact mode mattered more.

**Tradeoff:** this term should stay conservative. It helps keep commands
reasonable, but it cannot remove chatter or contact exploits by itself.

---

## ADR-005: Reward Term - `xz_vel_penalty`

**Status:** Accepted

**Context:** The robot developed horizontal drift and eventually tipped. Because
Chrono is Y-up, horizontal ground-plane motion is X/Z, not X/Y.

**Decision:**

```python
xz_vel = trunk_lin_vel[[0, 2]]
xz_vel_penalty = 0.20 * mean(xz_vel**2)
```

**Why it worked:** it reduced ground-plane drift after the standing controller
was already viable.

**Tradeoff:** it does not directly prevent lean, vibration, or a foot from being
unloaded. It is a drift term, not a contact term.

---

## ADR-006: Reward Term - `ang_vel_penalty`

**Status:** Accepted at low weight

**Context:** Several failed policies had significant trunk angular motion before
tip or height termination. Penalizing angular velocity gives early pressure
against body wobble.

**Decision:**

```python
ang_vel_penalty = 0.01 * mean(trunk_ang_vel**2)
```

**What did not work:** increasing angular-velocity weight too far made the robot
too constrained. It reduced some motion but could cause quiet tilting or falling.

**Tradeoff:** angular velocity is a damping signal, not the main standing
objective.

---

## ADR-007: Reward Term - `joint_vel_penalty`

**Status:** Accepted

**Context:** Visible shuffling remained even when pose error was tiny. A pose
penalty only sees displacement from home, not rapid motion around home.

**Decision:**

```python
joint_vel_penalty = 0.01 * mean(joint_vel**2)
```

**What worked:** joint velocity penalty reduced some oscillation and improved
smoothness metrics.

**What did not work:** it did not fix the lifted-foot stance. Diagnostics later
showed that issue was contact/support, not just velocity.

**Tradeoff:** this term makes the policy calmer, but too much could prevent fast
recovery motions during perturbations, rough terrain, or walking.

---

## ADR-008: Reward Term - `action_rate_penalty`

**Status:** Accepted at 0.03

**Context:** Joint-velocity penalty measures physical leg motion. It does not
directly punish twitchy target changes from the policy.

**Decision:**

```python
action_delta = clipped_action - prev_action
action_rate_penalty = 0.03 * mean(action_delta**2)
```

`reset()` clears `prev_action` to zeros.

**What worked:** action rate reduced turning and large target jumps. It helped
the policy look less twitchy.

**What did not work:** action rate did not solve the raised-foot stance. The
policy could be smooth and still choose a bad support pattern.

**Tradeoff:** rate penalties can make a policy slower to react. Keep watching it
when perturbations, friction randomization, and walking are added.

---

## ADR-009: Reward Term - `tilt_penalty`

**Status:** Accepted

**Context:** The robot survived but sometimes settled into a biased lean.
Raising upright reward was too blunt and brought back more shuffling.

**Decision:**

```python
tilt_error = trunk_x_up**2 + trunk_y_up**2
tilt_penalty = 0.25 * tilt_error
```

**Why it worked:** tilt directly punishes the non-up trunk axes having world-up
components. It fixed persistent lean better than simply raising upright reward.

**Tradeoff:** tilt penalty fixes lean, not contact support. It is still useful
after the contact fixes because it keeps the trunk centered.

---

## ADR-010: Diagnostic Term - `leg_symmetry_error`

**Status:** Diagnostic only

**Context:** The policy appeared to tuck and load one side differently. A weak
leg symmetry penalty was tested because it is more terrain-safe than absolute
foot-height balance.

**Tested decision:**

```python
leg_symmetry_error = 0.5 * (
    mean((FR_q - FL_q)**2) + mean((RR_q - RL_q)**2)
)
tested_penalty = weight * leg_symmetry_error
```

**What happened:**

- `0.05` was a weak guardrail but did not fix the lifted-foot stance.
- `0.20`, `0.25`, and `0.30` did not force clean contact and increased wobble.
- After disabling non-foot collisions and adding foot support, the symmetry
  error naturally dropped near `0.0017`.
- During friction B widening, the policy again found a persistent asymmetric
  load pattern while keeping non-foot contact at zero.
- A later `0.02` friction B experiment survived but did not remove the visible
  lean and made the contact signature worse than the clean AB reference.

**Decision:** keep `leg_symmetry_error` in logs and viewer diagnostics, but do
not include it in the reward while diagnosing friction B tilt.

**Tradeoff:** symmetry can fight valid corrective asymmetries if it is too
strong. The current tilt investigation should first identify whether the cause
is contact load, slip, policy action bias, or curriculum range before changing
reward terms again.

---

## ADR-011: Reward Term - `foot_contact_penalty`

**Status:** Accepted

**Context:** After hip/thigh/calf collisions were disabled, the old support
exploit disappeared, but the policy learned a different bad solution: stand
with one foot unloaded. The viewer showed `FL foot_load=0.0` even though the
robot survived.

**Decision:** require every foot to carry at least a small amount of load:

```python
foot_loads = [abs(foot.GetContactForce().y) for foot in feet]
missing_contact = max(0.0, MIN_FOOT_LOAD - foot_load) / MIN_FOOT_LOAD
foot_contact_error = mean(missing_contact**2)
foot_contact_penalty = 0.10 * foot_contact_error
```

Current threshold:

```text
MIN_FOOT_LOAD = 20 N
```

**Why it worked:** it attacks the real failure directly. The policy no longer
gets full credit for standing while one foot is floating or unused.

**Why this is better than foot-height reward:** SCM terrain can be uneven and
deformable. Feet can legitimately sit at different world Y heights. Contact load
is a better standing support signal than absolute foot height.

**Tradeoff:** contact-force rewards depend on the contact model, material, and
soil behavior. The threshold is deliberately low so it means "participate in
support," not "perfect equal load sharing."

---

## ADR-012: Home Pose Baseline - Less-Crouched Stance

**Status:** Accepted

**Context:** Zero action means "hold the position-control home pose." The
original Menagerie pose (`hip=0.0`, `thigh=0.9`, `calf=-1.8`) looked plausible
but sank under zero action in Chrono.

**Decision:**

```python
_HOME_JOINT_ANGLES = np.tile([0.0, 0.7, -1.4], 4)
_SPAWN_HEIGHT = 0.34
```

**Evidence from zero-action `view_env.py`:**

```text
less_crouched @ 0.32: y 0.320 -> 0.400 -> settles near 0.341
less_crouched @ 0.34: y 0.340 -> 0.341 and stays there
sbel_sign_adjusted @ 0.34: y 0.340 -> settles near 0.317
sbel_sign_adjusted @ 0.36: y 0.360 -> settles near 0.314
```

**Tradeoffs:**

- It departs from the exact MuJoCo Menagerie keyframe.
- It remains inside Go1 joint limits and is mechanically stable in this Chrono
  import.
- Zero-action standing does not make the policy pointless. The policy becomes a
  correction controller for drift, friction changes, terrain variation, and
  eventually SCM soil response.

---

## ADR-013: Action Scale

**Status:** Accepted at 0.20

**Context:** With `0.25`, the policy had enough authority but often moved feet
too much while balancing. Reducing the scale was tested after contact diagnostics
showed the issue was not only reward shaping.

**Decision:**

```python
target = home + 0.20 * clipped_action
```

**What worked:** smaller normalized offsets reduced stance excursions and made
the final foot-contact policy calmer.

**Tradeoff:** too small an action scale could limit recovery and walking later.
For standing, 0.20 is a good baseline. Revisit for walking if the policy cannot
step far enough.

---

## ADR-014: Joint Observation Source - Motor Frames

**Status:** Accepted

**Context:** The linked-body joint-angle workaround read thigh/calf joints near
zero at reset even though `DoAssembly()` placed the motors at home. This created
large false pose error before the robot moved.

**Decision:** Read joint position from the motor frames:

```python
frame1 = motor.GetFrame1Abs()
frame2 = motor.GetFrame2Abs()
q_rel = frame1.GetRot().GetInverse() * frame2.GetRot()
joint_angle = sign * q_rel.GetRotVec()[axis_idx]
```

**Result:** reset diagnostics report joint angles matching the home pose with
mean-squared error near zero.

**Consequence:** any policy trained before this fix is invalid for pose-reward
tuning.

---

## ADR-015: Tip Termination Threshold

**Status:** Accepted for current standing stage

**Context:** A very strict tip threshold terminated episodes before the robot had
room to learn recovery. A too-loose threshold would allow visibly fallen poses.

**Decision:**

```python
_MIN_UPRIGHT_ALIGNMENT = 0.85
```

**Tradeoff:** this gives PPO more recovery data. Evaluation still tracks
`min_upright_score`, so a policy that exploits the looser threshold is visible.
The current v2 policy stays far above the threshold.

---

## Diagnostics Added During This Phase

`view_stand_policy.py` is intentionally richer than `evaluate_stand.py`. The
viewer answers "what is physically happening right now?" while evaluation
summarizes episode-level metrics.

Current viewer diagnostics:

```text
height, upright, xz_vel
act_mean, dact_mean
jvel_mean
sym, tilt, pose_err
foot_dxz_mean/max
foot_vxz_mean/max
load_imb
foot_y
foot_share
foot_dxz
foot_load
calf_load, thigh_load, hip_load
leg_nonfoot_load
foot_y_min/max over interval
foot_load_min/max over interval
nonfoot_load_max over interval
```

Important interpretation rules:

- `foot_dxz` growing while `xz_vel` is small means feet are sliding/skittering
  under the body.
- `dact_mean` high means the policy target is twitching.
- `jvel_mean` high with low `dact_mean` points toward motor/contact dynamics.
- `calf/thigh/hip load` above zero means the robot is using non-foot terrain
  support.
- `foot_load=0` on one foot means survival is not enough; support must be
  rewarded directly.

Final accepted viewer signature:

```text
nonfoot_load_max = 0 on every leg
all four feet near the same world Y
no foot permanently at zero load
foot_dxz_mean around 0.015 after settling
upright around 1.000
no visible vibration
```

---

## Termination Conditions

```text
trunk_y < 0.22          fallen/collapsed too low on flat ground
upright_score < 0.85    tipped beyond current recovery-training range
obs contains NaN/Inf    physics solver exploded
step_count >= max_steps successful truncation
```

---

## Evaluation Checklist

After each retrain:

```text
survival_rate = 1.0
mean_length = 1000
min_trunk_y > 0.22
min_upright_score > 0.99 for the standing baseline
mean_abs_xz_vel low/stable
mean_abs_joint_vel low/stable
mean_abs_action_delta not exploding
mean_abs_action below saturation
min_foot_load not stuck at zero
foot_contact_error low/stable
termination_reasons = {'truncated': episodes}
```

Viewer diagnostics to watch:

```text
h / up                                  trunk height and upright score
xz                                      horizontal drift velocity
act / dact                              command size and target twitchiness
jvel                                    physical leg chatter
tilt                                    trunk lean
foot_min / load_imb                     foot participation and load balance
slip / vfoot                            foot displacement and skitter
nonfoot_max                             hidden calf/thigh/hip support
```

`view_stand_policy.py` keeps the detailed per-foot fields behind
`--full-diagnostics`. The default line was shortened after friction A because
the full diagnostic wall made it harder to see the acceptance signals while the
viewer was running.

Headless tilt diagnosis:

```bash
python diagnose_policy.py POLICY.zip --terrain flat --friction-min 0.6 --friction-max 1.0 --episodes 30 --out diagnostics/RUN_NAME
```

Use this before changing reward or physics when a policy survives but leans.
It reports whether foot unload, load imbalance, foot slip, action bias, or joint
asymmetry appears before tilt. Generated `diagnostics/` outputs are local and
gitignored.

---

## Commands

Train:

```bash
python train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000
```

Evaluate:

```bash
python evaluate_stand.py runs/stand/final_model.zip
```

View:

```bash
python view_stand_policy.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8
```

---

## Roadmap

```text
Stage 1  train_stand.py       flat terrain, fixed friction=0.8       <- accepted
Stage 2  train_stand.py       flat terrain, randomized friction A    <- accepted
Stage 2b train_stand.py       flat terrain, randomized friction B    <- accepted via AB generalization
Stage 2c train_stand.py       flat terrain, randomized friction C    <- next
Stage 3  train_walk.py        flat terrain walking
Stage 4  train_walk_scm.py    SCM deformable terrain fine-tuning
Stage 5  rollout collection   learned standing/walking skills
Stage 6  world model          obs/action/next_obs prediction
Stage 7  hierarchy            skill selection and planning
```

Next stage fine-tune command:

```bash
python friction_curriculum.py train friction_c --run
```

## Stage 2: Randomized Friction Curriculum

Current status:

```text
base fixed 0.8              accepted
friction_a 0.7-0.9          accepted, 300k continuation
friction_ab 0.65-0.95       accepted as B-capable checkpoint
friction_b 0.6-1.0          accepted by AB eval/view/diagnostics
friction_c 0.5-1.1          pending
```

What changed:

- Added staged friction randomization after the fixed-friction standing v2
  baseline.
- Kept reward, contact, collision, home pose, action scale, and solver settings
  unchanged so friction robustness is isolated from physics/reward changes.
- Forced SB3 PPO to CPU because this MLP policy and Chrono rollout loop do not
  benefit from CUDA in the current setup.
- Centralized accepted baseline paths, friction run directories, viewer
  defaults, and the SB3 device in `project_config.py` to reduce drift between
  scripts and docs.
- Promoted the AB checkpoint to `CURRENT_BASELINE_MODEL` after it passed the
  full B range, so viewers and future curriculum stages use the best accepted
  standing policy by default.

Why 300k was accepted for friction A:

- A 150k continuation survived but visually shuffled in place and leaned.
- The 300k continuation was visibly cleaner and matched the contact metrics:
  much lower foot-contact error, higher minimum foot load, lower action-rate
  motion, lower angular velocity, and lower X/Z motion.

Training-length lesson:

- Fine-tuning can use more steps than the original run. It means continuing
  from useful weights, not automatically using a tiny number of updates.
- More PPO steps are not accepted blindly. Each stage must still pass randomized
  eval, fixed-0.8 regression, viewer stability, foot contact, and zero non-foot
  support.

Why AB was accepted as B-capable:

- The AB checkpoint was trained on `0.65-0.95`, but evaluation on the full
  `0.6-1.0` range stayed clean: 30/30 truncated episodes, `min_upright_score`
  `0.999`, `foot_contact_error` near `0.0096`, and minimum foot load above
  `23 N`.
- Viewer inspection on `0.6-1.0` showed stable standing with no visible lean or
  vibration.
- Headless diagnosis showed no tilt-threshold crossing in 30/30 episodes on
  the full B range.
- Continued B training from AB created a repeatable FL-heavy left/right load
  bias before tilt. That makes extra B fine-tuning a regression, not a required
  curriculum step.
- `friction_curriculum.py train friction_b` is intentionally disabled; B eval
  and view commands use the AB checkpoint, and friction C training loads from
  AB.
- The rejected local B/ABB experiment folders were deleted after their failure
  signatures were documented. Accepted checkpoints remain local/out-of-band
  because `runs/` is gitignored.
