# Training Roadmap

Architecture Decision Records for the Chrono Go1 standing policy. Each section
is the context that forced a decision, what was decided, and what it costs.
Git history records what was tried. These docs record only the current rational state.

## Stage 1: Standing — Flat Terrain, Fixed Friction=0.8

```python
Go1Env(terrain="flat", enable_motors=True, friction_range=(0.8, 0.8))
```

Active reward: **reset — see ADR-008.** Building from scratch with a single term.

```python
reward = alive_bonus(+1.0)   # starting baseline — one term at a time
```

---

## ADR-001: Reward Term 1 — `alive_bonus`

**Status:** Superseded by ADR-008 (reward reset)

**Context:** The original first term was `height_score = clip((trunk_y − 0.15) / (0.28 − 0.15), 0, 1)`.
It encodes a Y=0 ground assumption. On SCM deformable terrain the ground surface
varies — a robot sinking correctly into soft soil would be penalised. Not transferable.

**Decision:** Replace with `alive_bonus = 1.0` per surviving step. Terrain-agnostic.
Matches MuJoCo baseline.

**Consequences:**
- Alive bonus does not gradient-push the robot to stand tall — upright score and
  termination floor do that job.
- `_TERM_HEIGHT = 0.15` is kept for termination only; will need a terrain-relative
  version for SCM.

---

## ADR-002: Reward Term 2 — `upright_score`

**Status:** Superseded by ADR-008 (reward reset)

**Context:** With only `alive_bonus`, the policy learned to pitch forward slowly.
Trunk height stayed above the termination floor but the robot gradually tipped.

**Decision:** `upright_score = max(0, trunk_local_Z · world_Y)`. Rewards keeping the
trunk's local Z axis (URDF "up") aligned with Chrono world Y-up. Termination
threshold `_MIN_UPRIGHT_ALIGNMENT = 0.75` added alongside.

**Consequences:**
- `Rotate(ChVector3d(0,0,1)).y` is the correct computation in this Y-up world.
- Termination is required alongside reward — reward alone was insufficient to stop
  a slowly tipping robot.

---

## ADR-003: Reward Term 3 — `pose_penalty` (weight=0.15)

**Status:** Superseded by ADR-008 (reward reset)

**Context:** After upright score was added, the robot found a new loophole: hold the
trunk level while legs fold straight down. Trunk sank gradually while `upright_score`
stayed at 1.0. Diagnostic data from a 1500-step episode:

```text
step 500:  trunk_y=0.353  (settled at home height)
step 900:  trunk_y=0.291  (sinking −0.014 per 100 steps)
step 1400: trunk_y=0.225  (approaching termination floor)
```

**Decision:** `pose_penalty = 0.15 × Σ(joint_angle − home_angle)²`. Penalises
deviation from the 12-joint home pose `[0.0, 0.9, −1.8]` per leg.

**Consequences:**
- Weight 0.15 is conservative. If trunk_y still drifts after retrain, raise to 0.2.
  Never jump to 0.4 — that weight was tested and caused policy collapse (see note below).
- _Rejected: weight=0.1_ — insufficient to hold height long-term.
- _Rejected: weight=0.4_ — caused penalty overload (see below).

**Penalty overload rule:** An earlier attempt added five penalty terms simultaneously
(`foot_height`, `action_rate`, `horizontal_vel`, `pose_weight=0.4`, `control_penalty`).
Result: mean_reward ≈ −3000, 0% survival. The reward had no reachable positive region.
**Rule: one term per training run. Evaluate before adding the next.**

---

## ADR-004: Reward Term 4 — `control_penalty`

**Status:** Superseded by ADR-008 (reward reset)

**Context:** After pose penalty was added, evaluation showed `max_abs_action = 1.000` —
policy saturated all joint targets to maximum offsets, causing jerky motion.

**Decision:** `control_penalty = 0.01 × Σ(action²)`. Matches MuJoCo baseline weight.

**Consequences:**
- Weight 0.01 is small enough not to compete with alive bonus or upright score.
- Complements pose_penalty: both discourage large joint offsets.

---

## ADR-005: Reward Term 5 — `ang_vel_penalty`

**Status:** Superseded by ADR-008 (reward reset)

**Context:** Trunk angular velocity was high in evaluation — robot wobbling rotationally
while surviving. `obs[10:13]` = trunk angular velocity in world frame.

**Decision:** `ang_vel_penalty = 0.05 × Σ(obs[10:13]²)`. Matches MuJoCo baseline weight.

**Consequences:**
- Weight 0.05 is moderate. Too heavy too early prevents posture recovery after perturbation.

---

## ADR-006: Joint Axis Reading — Hip Angles

**Status:** Accepted

**Context:** After pose_penalty was active, evaluation showed 100% lateral tip
termination (`trunk_y_up = −0.664`). Root cause: `_joint_angle()` always read the
Z component of the rotation vector. Hip joints rotate about URDF X — their Z component
is always ~0 regardless of actual abduction. Hips were invisible to the pose penalty.

**Decision:** Per-joint axis map and sign correction:

```python
_JOINT_AXES      = np.array([0,2,2, 0,2,2, 0,2,2, 0,2,2], dtype=np.int32)
_JOINT_AXIS_SIGN = np.where(_JOINT_AXES == 0, 1.0, -1.0)
```

Index 0 = X (hips), index 2 = Z (thigh/calf). Sign −1 for thigh/calf: URDF Y →
Chrono −Z after spawn rotation, so `GetRotVec().z = −θ`. Multiplying by −1 gives
+θ, matching home angles. Full geometric derivation: [chrono_port_notes.md](chrono_port_notes.md).

**Consequences:**
- Full retrain required — hip observation values changed from ~0 to real angles.
- _Rejected: sign=+1_ — at home pose, reading = −0.9 vs target = +0.9 →
  pose_error = 3.24 per joint → pose_penalty ≈ 6.5/step → −6500/episode → 0% survival.

---

## ADR-007: Home-Pose Spawn — `DoAssembly(POSITION)`

**Status:** Accepted

**Context:** `ChParserURDF.SetRootInitPose()` initialises only the root body. All joint
angles start at 0 — legs nearly straight down. The original fix was spawning at y=0.48
and ramping motor targets from 0 to home over 500 steps, wasting 500 training steps
per episode with no useful policy signal.

**Decision:** Initialise `ChFunctionConst` motors to home angles, then run Chrono's
kinematic assembly solver before the first `DoStepDynamics()`:

```python
self._trunk.SetFixed(True)
system.DoAssembly(1)   # pure constraint satisfaction — no forces, no time integration
self._trunk.SetFixed(False)
```

Spawn height `_SPAWN_HEIGHT = 0.27` (Menagerie equilibrium). Feet land at y≈0 after
assembly — no free-fall, no impact pitch. Zero compute overhead.

**Consequences:**
- Full retrain required.
- _Rejected: motor ramp (500 steps)_ — wasted training signal, complicated diagnostics.
- _Rejected: warm-up loop (50–200 steps)_ — worked but ~5% overhead per episode.
- _Rejected: spawn at 0.35 m_ — 8 cm free-fall caused forward pitch on impact.
- _Inspiration:_ [harryzhang1018](https://github.com/harryzhang1018) /
  [SBEL multi-terrain RL](https://github.com/uwsbel/sbel-reproducibility/tree/master/2025/multi-terrain-RL)
  (UW-Madison, 2025) — calls `actuate(home)` before any physics steps. Our `DoAssembly`
  achieves the same goal without a warm-up loop.

---

## ADR-008: Reward System Reset

**Status:** Active

**Context:** ADR-001 through ADR-005 were developed under compounding incorrect
conditions:
- The hip axis bug (ADR-006) was active during early training runs — hips were
  invisible to the pose penalty, so policies that appeared to converge were not
  actually holding the home pose.
- The sign-flip mistake (sign=+1 for thigh/calf) was introduced and reverted,
  invalidating any checkpoint trained during that period.
- Penalty overload (five terms added simultaneously, pose_weight=0.4) caused 0%
  survival and led to incorrect weight choices.
- The spawn height was 0.35 m → 8 cm free-fall → forward pitch on impact,
  meaning all previous training ran with a systematically biased initial condition.
- No policy trained under the previous reward currently meets the evaluation
  criteria (`survival_rate=1.0`, `mean_reward≈900`).

**Decision:** Wipe the reward. Start from a single term and add one term per training
run, with a full evaluation pass before adding the next. Termination conditions
(height + upright angle + NaN guard) are unchanged.

Starting baseline:

```python
reward = alive_bonus   # +1.0 per surviving step
```

**Why `alive_bonus` and not a height score:**
A height score (`clip((trunk_y − 0.15) / (0.28 − 0.15), 0, 1)`) encodes the
assumption that the ground surface is at Y=0. On SCM deformable terrain the
ground varies — a robot correctly holding its joints while sinking 5 cm into
soft soil would be penalised even though it is doing the right thing. `alive_bonus`
is terrain-agnostic: any step that does not meet a termination condition earns
+1.0, regardless of absolute trunk height. The termination floor (`trunk_y < 0.15`)
still needs the Y=0 assumption on flat terrain — that threshold will need a
terrain-relative version for SCM.

Evaluation gate before adding any next term:

```text
survival_rate = 1.0
mean_reward ≈ 1000
trunk_y stable from step 0 (no pitch on spawn)
```

**Consequences:**
- ADR-001 through ADR-005 are superseded. Their context and reasoning are preserved
  as the record of what was tried. Git history preserves the code.
- All previous policy checkpoints are invalid — retrain from scratch.
- The "one term at a time" rule (from the penalty overload failure) is the core
  constraint going forward.

---

---

## Termination Conditions

```text
trunk_y < 0.15          — fallen to ground
upright_score < 0.75    — tipped more than ~41°
obs contains NaN/Inf    — physics solver exploded
```

---

## Evaluation Checklist

After each retrain:

```text
survival_rate = 1.0       (100% of episodes complete full 1000 steps)
mean_reward ≈ 900–1000    (1.0 alive × 1000 steps − small penalties)
mean_abs_action < 0.5     (not saturating)
max_abs_action < 0.9      (well away from clip limits)
ang_vel low               (trunk not spinning)
trunk_y stable from step 0
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

Evaluation targets:

```text
survival_rate     — 1.0
mean_reward       — ≈ 900–1000
mean_length       — 1000
mean_abs_action   — < 0.5
max_abs_action    — < 0.9
min_trunk_y       — > 0.25
min_upright_score — > 0.9
```

---

## Roadmap

```text
Stage 1  train_stand.py     flat terrain, fixed friction=0.8      ← active
           ↳ position_penalty (×0.5)
           ↳ xy_vel_penalty (×0.1)
Stage 2  train_stand.py     flat terrain, friction randomized (0.6–1.0)
Stage 3  train_walk.py      flat terrain walking
Stage 4  train_walk_scm.py  SCM deformable terrain fine-tuning
Stage 5  rollout collection learned standing/walking skills
Stage 6  world model        obs/action/next_obs prediction
Stage 7  hierarchy          skill selection and planning
```

Stage 2 fine-tune command:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe train_stand.py --terrain flat --friction-min 0.6 --friction-max 1.0 --load runs/stand/final_model.zip --save-dir runs/stand_friction_narrow
```
