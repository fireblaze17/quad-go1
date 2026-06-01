# Training Roadmap

This file records the current rational path through the standing experiments.
Git history has the exact old code states; this document explains what we now
believe and what should happen next.

## Current Direction

The project is back in randomized-friction standing cleanup.

AB remains the best archived checkpoint:

```text
runs/stand_friction_ab_065_095/final_model.zip
```

But AB is no longer considered final-clean standing. It passed the older
criteria on B/C friction ranges, then later settled-window diagnostics showed
that drift/slip is already present before reset-noise is introduced. Therefore
reset-noise training is paused. The next useful work is to rebuild a cleaner
AB-style policy from the friction curriculum.

## Active Runtime Reward

The active code is restored to the AB/friction-era reward behavior:

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
```

Removed from active runtime:

```text
reset-noise profiles
clean-standing bonus
direct load-balance reward experiments
max-share and left/right load penalties
foot-slip reward experiments
contact-switch reward pressure
reset-noise clean/noisy comparison mode
```

Generic PPO knobs remain in `train_stand.py` because they are useful outside
reset noise:

```text
--learning-rate
--clip-range
--target-kl
```

## Why AB Passed Before

AB was trained as a continuation from friction A:

```text
base fixed friction -> friction A 0.7-0.9 -> AB 0.65-0.95
```

It passed the older checks:

```text
100/100 survival on C range 0.5-1.1
full 1000-step episodes
high upright score
viewer looked good enough
no non-foot collision exploit
better than C-from-AB and scratch-C challengers
```

That made AB the correct old baseline. The mistake would be treating survival
as final proof of clean standing.

## Why AB Must Be Redone

New diagnostics added after the reset-noise phase split behavior into full,
early, and settled windows. Those diagnostics showed that the standing problem
is not only "stay alive." It is also:

```text
stay upright
stay in place
keep all feet planted cleanly
avoid creeping/slipping
avoid biased support patterns
```

AB is still useful as an archived comparison point, but it already has enough
settled drift/slip that reset-noise training starts from a shaky foundation.
Reset noise made the failure obvious; it did not create the root issue.

## Reset-Noise Archive

Reset-noise runtime support and reset-noise reward experiments are intentionally
removed from active code. The main lesson is kept here:

```text
reward shaping during reset-noise training often improved one metric while
worsening another, especially drift, contact switching, and load bias.
```

The reset-noise branch produced useful diagnostics and one archived reference
run:

```text
runs/stand_reset_noise_a_slip0005_fullc_from50k_25k/
```

That run is not an active baseline. It is kept only as evidence for what was
tried.

## Current Diagnostic Workflow

Use settled-window diagnostics before accepting a standing candidate:

```bash
python run_regression.py POLICY.zip --name RUN_NAME --friction-min 0.5 --friction-max 1.1 --episodes 100
python compare_friction_slices.py --baseline runs/stand_friction_ab_065_095/final_model.zip --candidate POLICY.zip --name RUN_NAME_slices --episodes 100
python nominal_load_sanity.py --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10
python view_stand_policy.py POLICY.zip --terrain flat --friction-min 0.5 --friction-max 1.1
```

Important acceptance signals:

```text
survival_rate
mean_abs_xz_vel
settled_mean_abs_xz_vel
base_xz_displacement
signed trunk_x_up/trunk_y_up
min_foot_load
foot_contact_error
foot load shares
contact switches
contact-conditioned slip distance
viewer drift/tilt/sliding
```

## Next Experiment

Start before AB and train a cleaner randomized-friction replacement:

```bash
python train_stand.py \
  --terrain flat \
  --friction-min 0.65 \
  --friction-max 0.95 \
  --load runs/stand_friction_a_07_09/final_model.zip \
  --save-dir runs/stand_friction_ab_clean_retry \
  --timesteps 300000 \
  --seed 1 \
  --checkpoint-freq 50000
```

Evaluate checkpoints, not only the final model. The best standing checkpoint may
appear before the end of training.

## Acceptance Rule

A new AB replacement must beat archived AB on settled quality, not merely match
survival:

```text
100/100 survival on C range
lower settled drift than AB
lower or comparable contact error
healthy min foot load
no persistent viewer sliding
no obvious biased support pattern
no non-foot collision load
```

Only after this randomized-friction standing policy is clean should reset-noise
support be reintroduced.
