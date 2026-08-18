# Documentation

This document records the current working Go1 locomotion stack and the debugging path that made it work in Project Chrono. It is written as the main project reference: use it to understand what the default stack is, why the current choices were made, and what should be rechecked before changing the simulator, actuator, reward, or training setup.

## Current Baseline

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

This neutral checkpoint currently contains the 571M-step SCM-fine-tuned policy.

The default model is a trained locomotion policy, not a standing-only policy. The original training artifact is kept in `runs/` as a backup, but active code and docs use the neutral checkpoint copy above. Because run artifacts and model zips are ignored, a fresh checkout must be given this file out of band before the viewer or diagnostics can load the baseline.

Current stack:

```text
observation: 48D
actuator: actuator_net
alternate actuator: torque_limited_pd
contact: NSC flat rigid ground
ground friction: 0.8
policy rate: 50 Hz
physics rate: 200 Hz
physics substeps: 4
episode length: 1000 policy steps
action bounds: [-100, 100]
action scale: 0.25 rad
reward clipping: off
```

Home pose:

```text
FR [-0.05, 0.75, -1.30]
FL [ 0.05, 0.75, -1.30]
RR [-0.05, 0.85, -1.30]
RL [ 0.05, 0.85, -1.30]
```

The command sampler is:

```text
10% zero command:
  vx = 0.0, vz = 0.0, yaw_rate = 0.0

90% moving command:
  vx       ~ Uniform(-1.0, 1.0)
  vz       ~ Uniform(-0.6, 0.6)
  yaw_rate ~ Uniform(-1.0, 1.0)
```

Commands are body-frame commands. Chrono is Y-up in this project, so planar velocity is body X/Z and yaw is about Y.

## URDF And Collision Work

Collision geometry had to be debugged before reward or policy behavior could be trusted. Several early policies appeared to stand, slide, or move because collision bodies were supporting the robot in ways the visual mesh did not make obvious.

What was learned:

- Visual meshes are not enough for debugging; the viewer needs collision-only mode because the render mesh can look reasonable while collision boxes are wrong.
- Thigh, calf, foot, and trunk contact diagnostics remain available so non-foot support can be detected even when the active reward does not penalize it.
- Self-collision filtering is needed to prevent neighboring robot links from fighting each other.
- Robot-ground contacts are kept for trunk, feet, thighs, and calves so diagnostics can reveal unintended support or strikes.
- Collision-box length and placement had to be checked against the foot spheres and link visuals; otherwise calf/thigh bodies could report contact that was hard to see visually.
- The current active reward does not directly penalize thigh/calf contact. These contacts are still logged, because collision artifacts previously changed policy behavior.

Useful collision checks:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --visual-mesh-format none \
  --show-collision-boxes
```

## Solver And Contact Work

The active contact setup is NSC flat rigid ground with fixed friction `0.8`. Contact settings are part of the learned policy's world; changing them can invalidate behavior even when the policy and reward are unchanged.

What was learned:

- Soft-contact tests were useful for understanding penetration and support forces, but the robot could sag, penetrate, or explode when stiffness, damping, timestep, and solver settings were not matched carefully.
- Solver-iteration changes altered standing stability and contact loads, so solver settings had to be treated as stack parameters rather than harmless performance knobs.
- NSC became the reliable default for the current trained baseline.
- The flat ground is large enough for default 1000-step episodes across the command range.
- Friction is fixed at `0.8`; the foot material friction is higher so the ground value determines effective contact friction.

## Deformable Terrain Work

CRM/SPH terrain was tested as a higher-fidelity deformable-terrain path, but it is too computationally heavy for practical fine-tuning on this setup. The active deformable-terrain backend is SCM.

SCM keeps the policy interface aligned with the flat baseline:

```text
observation: 48D
action bounds: [-100, 100]
actuator: actuator_net by default
policy rate: 50 Hz
physics rate: 200 Hz
physics substeps: 4
reward: current default reward
```

The SCM backend uses Chrono's SCM terrain in the same Y-up project frame:

```text
system: ChSystemSMC
gravity: (0, -9.81, 0)
solver: BARZILAIBORWEIN
solver iterations: 60
terrain size: 20 x 20
grid spacing: 0.02
Bekker Kphi: 3e6
Bekker Kc: 0
Bekker n: 1.1
cohesion: 0
friction angle: 30 deg
Janosi shear: 0
elastic stiffness: 2e9
damping: 3e4
```

SCM viewing uses the source-built Chrono VSG path:

```bash
python view_scm_policy_vsg.py \
  runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0 \
  --max-steps 500 \
  --render-fps 1000000 \
  --ignore-termination
```

## Home Pose Work

Zero-action standing was used as the first sanity check. The policy can still learn recovery, but a home pose that immediately collapses makes locomotion training spend early capacity on reset survival instead of commanded motion.

What was tried and learned:

- Front/rear thigh and calf sweeps showed that small sagittal changes can strongly affect rear loading, foot clearance, and drift.
- Hip-offset tests helped separate lateral stance effects from sagittal drift.
- Spawn-height tests showed that contact initialization matters, but height alone did not solve unstable poses.
- Collision-box inspection was necessary because some pose failures were contact-geometry failures, not only controller failures.
- Dynamic spawn-height ideas helped explain foot penetration, but the final stack uses the default reset behavior that matches the successful run.
- The final home pose was selected because it stood cleanly enough under zero action with the current actuator/contact setup and then supported long locomotion training.

## Actuator Work

Actuator behavior was one of the central project issues. Ideal position control was useful for proving that poses and commanded visual motion were possible, but it was not the final physical actuator.

Torque-limited PD remains as a debugging actuator:

```text
tau_raw = kp * (q_target - q) - kd * qdot
tau = clip(tau_raw, -effort_limit, effort_limit)
```

The default actuator is the learned actuator model:

```text
resources/actuator_nets/unitree_go1.pt
```

Actuator-net inputs are the current and previous two policy-step samples of:

```text
q - q_target
qdot
```

Final actuator details:

- The policy updates joint targets at 50 Hz.
- Chrono integrates physics at 200 Hz.
- The actuator history updates once per 50 Hz policy step and is held across the four physics substeps.
- Output torque is clipped by URDF effort limits before being applied through Chrono force motors.
- Updating actuator history at the wrong rate changes what the learned actuator sees and can cause jitter or weak response.
- The actuator net is the default because it produced the working locomotion baseline; `torque_limited_pd` remains for controlled comparisons only.

## Reward Work

The reward is computed once per policy step:

```text
reward = 0.02 * raw_reward
```

Positive reward clipping is off. Terminal reward is zero.

Active terms:

```text
tracking_lin_vel      +1.5
tracking_ang_vel      +0.75
lin_vel_z             -2.0
ang_vel_xy            -0.05
torques               -0.0002
dof_acc               -2.5e-7
flat_orientation_l2   -2.5
feet_air_time         +0.25
action_rate           -0.01
termination            0.0
```

Important reward lessons:

- Standing-only rewards created local minima and did not automatically become locomotion.
- Body-frame velocity tracking is the active task definition.
- World-frame velocity tracking can make a policy appear to work while solving the wrong problem.
- Chrono is Y-up here, so vertical velocity, yaw, and projected gravity calculations must use the correct axes.
- Positive reward clipping is off in the current baseline.
- Terminal reward is zero, so the policy is mainly shaped by per-step tracking and regularization.
- Feet air time is a gait-shaping term, not the main task reward.
- Action-rate, torque, and acceleration penalties are stabilizers; they should be read with velocity tracking and viewer behavior, not alone.
- Short early viewers can be misleading; the sampled policy often improves before the deterministic policy looks clean.

## Training Work

The current default training setup is:

```text
num_envs: 24
n_steps: 256
rollout_size: 6144
batch_size: 1536
n_epochs: 3
learning_rate: 1e-4
learning_rate_final: 1e-4
clip_range: 0.1
target_kl: 0.015
ent_coef: 0.001
gamma: 0.99
gae_lambda: 0.95
max_grad_norm: 1.0
checkpoint_freq: 1000000
```

Training lessons:

- With only 24 environments, useful behavior may take many millions of steps to appear.
- Episode reward, episode length, velocity tracking, policy standard deviation, KL, action-rate penalty, and torque penalty should be read together.
- A deterministic viewer may lag behind stochastic training behavior while the policy standard deviation is still high.
- Long runs were necessary to separate failed local minima from policies that were still improving.
- If reward rises while episode length falls, run fixed-command diagnostics before assuming the policy improved.
- If velocity tracking improves while contact metrics worsen, inspect the viewer and collision-only view before extending the run.
- Continuing a run should use both the saved policy and saved state when preserving timestep accounting matters.

## Randomization

The current randomization profile is part of the default stack:

```text
friction: fixed 0.8
joint reset multiplier: 1.0
root x/z offset: Uniform(-0.5, 0.5)
root yaw: Uniform(-pi, pi)
root linear velocity: zero
root angular velocity: zero
observation noise: enabled
base mass add: Uniform(-1.0, 3.0)
COM offset: disabled
pushes: disabled
```

Old standing-only friction and reset-noise experiments are historical and are not active locomotion guidance.

The current randomization is intentionally simple. Friction is fixed, pushes are disabled, and COM offset is disabled. The randomization that remains is the part used by the working baseline.

## Reproduction

View the baseline:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/default_baseline.zip
```

Evaluate fixed forward motion:

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

View collision shapes:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --visual-mesh-format none \
  --show-collision-boxes
```

For a clean checkout, place the baseline model at:

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

The file is intentionally not tracked by git.
