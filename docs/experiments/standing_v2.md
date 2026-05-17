# Standing V2 Model Card

## Summary

Flat-ground standing v2 is the accepted fixed-friction baseline for the Chrono
Go1 environment. It stands for the full 1000-step episode, keeps all four feet
near the terrain, avoids non-foot leg contact, and has no obvious vibration in
the viewer.

## Training Command

```bash
python train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000 --seed 1 --save-dir runs/stand
```

PPO settings in `train_stand.py`:

```text
policy:        MlpPolicy
n_steps:       1024
batch_size:    256
learning_rate: 3e-4
gamma:         0.99
seed:          1
```

## Environment Constants

```text
terrain:             flat
friction_range:      (0.8, 0.8)
max_steps:           1000
time_step:           0.002
home pose per leg:   [0.0, 0.7, -1.4]
spawn height:        0.34
action scale:        0.20
collision bodies:    trunk, FR_foot, FL_foot, RR_foot, RL_foot
solver:              BARZILAIBORWEIN
solver iterations:   60
ground contact:      friction=0.8, restitution=0.1, Kn=2e5, Gn=60
foot contact:        friction=0.9, restitution=0.01, Gn=60
minimum foot load:   20 N
```

## Reward Weights

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

`leg_symmetry_error` is logged for diagnosis, but is not used as an active
reward penalty in this accepted baseline.

## Evaluation Command

```bash
python evaluate_stand.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10
```

Accepted evaluation:

```text
episodes: 10
survival_rate: 1.000
mean_reward: 1129.088
mean_length: 1000.0
mean_abs_action: 0.304
max_abs_action: 1.000
min_trunk_y: 0.337
min_upright_score: 1.000
mean_abs_xz_vel: 0.007387
mean_abs_joint_vel: 0.393387
mean_foot_load: 32.091613
min_foot_load: 17.575732
foot_contact_error: 0.020161
termination_reasons: {'truncated': 10}
friction_min_seen: 0.800
friction_max_seen: 0.800
```

## Viewer Contact Signature

Use:

```bash
python view_stand_policy.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8
```

Expected signature after the first settling period:

```text
upright near 1.000
foot_y values close together
foot_load nonzero on all feet
calf_load=(FR:+0.0,FL:+0.0,RR:+0.0,RL:+0.0)
thigh_load=(FR:+0.0,FL:+0.0,RR:+0.0,RL:+0.0)
hip_load=(FR:+0.0,FL:+0.0,RR:+0.0,RL:+0.0)
nonfoot_load_max=(FR:+0.0,FL:+0.0,RR:+0.0,RL:+0.0)
```

## Known Limitations

- This is fixed-friction flat-ground standing, not robust standing.
- The policy is not yet trained on randomized friction.
- The policy has not yet been validated on SCM deformable terrain.
- Foot-contact reward magnitudes may need retuning on SCM if contact forces
  differ significantly.
- Thigh/calf/hip terrain collisions are disabled by design to prevent non-foot
  support exploits during locomotion learning.
