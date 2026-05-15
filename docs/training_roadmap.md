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
)
```

Current weights:

```text
alive_bonus             1.00
upright_reward          0.15 * upright_score
pose_penalty            0.10 * mean(joint_error^2)
control_penalty         0.03 * mean(action^2)
joint_vel_penalty       0.01 * mean(joint_velocity^2)
action_rate_penalty     0.01 * mean(action_delta^2)
tilt_penalty            0.25 * (trunk_x_up^2 + trunk_y_up^2)
angular_vel_penalty     0.01 * mean(trunk_angular_velocity^2)
xz_vel_penalty          0.20 * mean([vx, vz]^2)
```

Current evaluation checkpoint:

```text
survival_rate:       1.000
mean_length:         1000.0
min_trunk_y:         0.336
min_upright_score:   0.994
mean_abs_xz_vel:     0.040
mean_abs_joint_vel:  0.283
mean_abs_action:     0.380
termination:         truncated only
```

Conclusion: flat-ground standing v1 is accepted. It is not perfect: there is
still mild in-place leg chatter. But the policy stands upright, does not sink,
does not drift into a fall, and no longer settles into a large one-sided lean.

---

## ADR-001: Reward Term 1 - `alive_bonus`

**Status:** Accepted

**Context:** A base-height reward was considered first, but absolute height
assumes a flat Y=0 ground plane. That is fragile for Chrono SCM terrain, where
the foot-ground interface can sink or deform.

**Decision:** Use terrain-agnostic survival reward:

```python
alive_bonus = 1.0
```

**Why it worked:** it creates a simple base objective: remain in a valid standing
episode for 1000 steps. Height and tilt stay as termination conditions or
specialized penalties, not the main reward source.

**Tradeoff:** alive reward alone is sparse. It does not distinguish clean
standing from barely surviving, so shaping terms are still required.

---

## ADR-002: Reward Term 2 - `upright_reward`

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

**What did not work:** increasing upright weight to 0.25 reduced visible lean,
but it also brought back more leg shuffling. Upright reward is too broad: it
encourages the policy to correct the trunk, but does not say how to correct it.

**Tradeoff:** the accepted 0.15 weight is intentionally modest. A separate tilt
penalty now handles persistent lean more directly.

---

## ADR-003: Reward Term 3 - `pose_penalty`

**Status:** Accepted at low weight

**Context:** Early alive/upright policies could keep the trunk mostly vertical
while using odd leg configurations. Once the joint observation bug was fixed,
pose error became meaningful.

**Decision:**

```python
pose_error = joint_pos - home_joint_angles
pose_penalty = 0.10 * mean(pose_error**2)
```

**What worked:** pose penalty keeps the legs near the known stable Chrono home
stance.

**What did not work:** trying to solve later falling/shuffling by increasing
pose weight was the wrong diagnosis. Current failures showed tiny pose error,
so the legs were near home while the body drifted or leaned.

**Tradeoff:** pose penalty preserves Go1-like geometry, but too much pose
regularization can fight necessary balance corrections.

---

## ADR-004: Reward Term 4 - `control_penalty`

**Status:** Accepted at 0.03

**Context:** The policy often used saturated action targets. A control penalty
discourages large sustained commands.

**Decision:**

```python
control_penalty = 0.03 * mean(action**2)
```

**What did not work:** larger control weights reduced corrective authority and
made the robot tip or spin sooner. The problem was not simply "actions are too
large"; the timing and physical motion of the legs also mattered.

**Tradeoff:** this term should stay conservative. It helps keep commands
reasonable, but it cannot remove chatter by itself.

---

## ADR-005: Reward Term 5 - `xz_vel_penalty`

**Status:** Accepted

**Context:** The robot developed horizontal drift and eventually tipped. Because
Chrono is Y-up, horizontal ground-plane motion is X/Z, not X/Y.

**Decision:**

```python
xz_vel = trunk_lin_vel[[0, 2]]
xz_vel_penalty = 0.20 * mean(xz_vel**2)
```

**Why 0.20:** the old MuJoCo reference used a sum-style horizontal velocity
penalty. This code uses `mean`, so 0.20 over two axes gives a similar scale.

**What worked:** mean X/Z velocity dropped into the acceptable range around
0.04 in the flat-ground v1 checkpoint.

**Tradeoff:** it reduces ground-plane drift, but it does not directly prevent
lean or leg chatter.

---

## ADR-006: Reward Term 6 - `ang_vel_penalty`

**Status:** Accepted at low weight

**Context:** Several failed policies had significant trunk angular motion before
tip or height termination. Penalizing angular velocity gives early pressure
against body wobble.

**Decision:**

```python
ang_vel_penalty = 0.01 * mean(trunk_ang_vel**2)
```

**What did not work:** increasing angular-velocity weight too far made the robot
too constrained. It reduced motion but could also make the policy less able to
recover, causing quiet sinking or tipping.

**Tradeoff:** angular velocity is useful as a damping signal, not as the main
standing objective.

---

## ADR-007: Joint-Velocity Diagnostics

**Status:** Diagnostic accepted; later became ADR-008 reward term

**Context:** After the corrected home pose, the robot could survive but still
visibly shuffled. Pose error stayed low, so the legs were not drifting far from
home; they were likely moving quickly around home.

**Decision:** Add telemetry:

```python
mean_abs_joint_vel = mean(abs(joint_vel))
max_abs_joint_vel = max(abs(joint_vel))
```

`view_stand_policy.py` prints these as `jvel_mean` and `jvel_max`.

**Result:** the telemetry lined up with visible leg chatter, so joint velocity
became a real reward term.

**Tradeoff:** joint velocity still comes from the linked-body approximation while
joint position comes from motor frames. That is acceptable for the current
smoothness signal, but should be revisited if this term becomes central to later
walking.

---

## ADR-008: Reward Term 7 - `joint_vel_penalty`

**Status:** Accepted

**Context:** Visible shuffling remained even when pose error was tiny. A pose
penalty only sees displacement from home, not rapid motion around home.

**Decision:**

```python
joint_vel_penalty = 0.01 * mean(joint_vel**2)
```

**What worked:** joint velocity penalty reduced some oscillation and improved
the smoothness metrics.

**What did not fully work:** by itself, it did not eliminate shuffling. One run
also introduced yaw/turn bias, showing that smoothing the legs can change how
the body finds balance.

**Tradeoff:** this term makes the policy calmer, but too much could prevent fast
recovery motions later during perturbations, rough terrain, or walking.

---

## ADR-009: Reward Term 8 - `action_rate_penalty`

**Status:** Accepted for standing v1

**Context:** Joint-velocity penalty measures physical leg motion. It does not
directly punish twitchy target changes from the policy. The viewer showed
nontrivial action deltas during visible shuffling.

**Decision:** Store the previous clipped action in the env and penalize target
changes:

```python
action_delta = clipped_action - prev_action
action_rate_penalty = 0.01 * mean(action_delta**2)
```

`reset()` clears `prev_action` to zeros.

**What worked:** action delta dropped and the robot became smoother in the
accepted v1 run.

**Consequence:** the environment now has one step of action memory. That is
normal for rate penalties, but it means reset behavior must always clear the
previous action.

**Tradeoff:** rate penalties can make a policy slower to react. Keep this term
small until perturbation recovery is tested.

---

## ADR-010: Reward Term 9 - `tilt_penalty`

**Status:** Accepted

**Context:** After X/Z velocity, joint velocity, action rate, and low upright
reward, the robot survived but settled into a biased lean. Raising upright
reward helped lean but brought back more shuffling. Upright reward was too
blunt.

**Decision:** Add a direct trunk tilt penalty:

```python
tilt_error = trunk_x_up**2 + trunk_y_up**2
tilt_penalty = 0.25 * tilt_error
```

This is separate from `upright_score`. It directly punishes non-up trunk axes
having world-up components.

**Why it worked:** the earlier leaning runs had average `trunk_y_up` around
0.15 in one direction or the other. After adding tilt penalty, the accepted v1
run had:

```text
trunk_y_up mean:  -0.042
tilt_error mean:  0.0037
min_upright:      0.994
survival:         1.000
```

**Tradeoff:** tilt penalty fixes lean but does not directly remove leg chatter.
That is why v1 is stable but still not perfectly smooth.

---

## ADR-011: Home Pose Baseline - Less-Crouched Stance

**Status:** Accepted

**Context:** Zero action means "hold the position-control home pose." The
original Menagerie pose (`hip=0.0`, `thigh=0.9`, `calf=-1.8`) looked plausible
but sank under zero action in Chrono. Reward tuning was compensating for a bad
neutral pose.

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

`less_crouched @ 0.34` starts closest to its natural support height.

**Tradeoffs:**
- It departs from the exact MuJoCo Menagerie home keyframe.
- It remains inside Go1 joint limits and is mechanically stable in this Chrono
  import.
- Zero-action standing does not make the policy pointless. The policy becomes a
  correction controller for drift, friction changes, terrain variation, and
  eventually SCM soil response.

---

## ADR-012: Joint Observation Source - Motor Frames

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

## ADR-013: Tip Termination Threshold

**Status:** Accepted for current standing stage

**Context:** A very strict tip threshold terminated episodes before the robot had
room to learn recovery. A too-loose threshold would allow visibly fallen poses.

**Decision:**

```python
_MIN_UPRIGHT_ALIGNMENT = 0.85
```

**Tradeoff:** this gives PPO more recovery data. Evaluation still tracks
`min_upright_score`, so a policy that exploits the looser threshold is visible.
The current v1 policy stays far above the threshold (`min_upright_score=0.994`).

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
min_upright_score ideally > 0.95, v1 target > 0.99
mean_abs_xz_vel low/stable
mean_abs_joint_vel low/stable
mean_abs_action_delta low/stable
mean_abs_action below saturation
max_abs_action not constantly clipped
termination_reasons = {'truncated': episodes}
```

Viewer diagnostics to watch:

```text
axis=(trunk_x_up, trunk_y_up, trunk_z_up)  lean direction
ang_xyz                                  body angular velocity
xz_vel                                   ground-plane drift
dact_mean/dact_max                        target twitchiness
jvel_mean/jvel_max                        physical leg chatter
tilt                                     lean penalty input
pen=(...)                                reward term scale comparison
```

---

## Commands

Train:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000
```

Evaluate:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe evaluate_stand.py runs/stand/final_model.zip
```

View:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe view_stand_policy.py runs/stand/final_model.zip
```

---

## Roadmap

```text
Stage 1  train_stand.py       flat terrain, fixed friction=0.8
           -> flat-ground standing v1 accepted
           -> next: reduce mild in-place shuffling
Stage 2  train_stand.py       flat terrain, randomized friction
Stage 3  train_walk.py        flat terrain walking
Stage 4  train_walk_scm.py    SCM deformable terrain fine-tuning
Stage 5  rollout collection   learned standing/walking skills
Stage 6  world model          obs/action/next_obs prediction
Stage 7  hierarchy            skill selection and planning
```

Stage 2 fine-tune command:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe train_stand.py --terrain flat --friction-min 0.6 --friction-max 1.0 --load runs/stand/final_model.zip --save-dir runs/stand_friction_narrow
```
