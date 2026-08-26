# Unitree Go1 Locomotion in Project Chrono

This repository trains and evaluates a Unitree Go1 locomotion policy in Project Chrono / PyChrono. The policy is trained with PPO and tracks body-frame planar velocity plus yaw-rate commands. Control is actuator-network-driven torque control, with execution on both NSC rigid terrain and SCM deformable terrain.

## Policy Showcase

| NSC flat ground | SCM deformable terrain |
| --- | --- |
| <video src="media/nsc_baseline.mp4" controls width="420"></video> | <video src="media/scm_baseline.mp4" controls width="420"></video> |
| Flat-ground locomotion | Deformable-terrain locomotion |

Baseline policy:

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

This checkpoint path currently contains the 571M-step SCM-fine-tuned policy. Model zips are not tracked by git, so a fresh checkout needs the baseline artifact placed at this path before running viewer or diagnostic commands.

## Quick Start

Create the source-built Chrono environment:

```bash
conda env create -f environment.yml
conda activate chrono-src
bash scripts/setup_chrono_source.sh

conda deactivate
conda activate chrono-src
```

View the baseline on NSC rigid terrain:

```bash
python view_stand_policy.py
```

View the baseline on SCM deformable terrain:

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

Train a new default run:

```bash
python train_policy.py \
  --save-dir runs/default_new_run \
  --target-total-steps 100000000 \
  --checkpoint-freq 1000000
```

Open TensorBoard:

```bash
tensorboard --logdir runs/default_new_run/tensorboard --port 6006
```

## System at a Glance

| Component | Configuration |
| --- | --- |
| Robot | Unitree Go1 |
| Simulator | Project Chrono / PyChrono |
| RL algorithm | PPO |
| Default actuator | `actuator_net` |
| Actuator model | `resources/actuator_nets/unitree_go1.pt` |
| Policy rate | 50 Hz |
| Physics rate | 200 Hz |
| Physics substeps | 4 |
| Observation space | 48D |
| Action space | 12D continuous |
| Command space | `[vx, vz, yaw_rate]` |
| Default backend | NSC rigid ground |
| Optional backend | SCM deformable terrain |
| Episode length | 1000 policy steps |

## Coordinate and Command Convention

- Chrono uses a Y-up frame.
- X/Z are the planar motion axes.
- Y is vertical.
- Yaw is rotation about Y.
- Commands are expressed in the robot body frame.

| Command mode | Probability | Sampling |
| --- | ---: | --- |
| Standing | 10% | `vx = 0`, `vz = 0`, `yaw_rate = 0` |
| Moving | 90% | `vx ~ U(-1.0, 1.0)`, `vz ~ U(-0.6, 0.6)`, `yaw_rate ~ U(-1.0, 1.0)` |

## Policy Interface

Observation layout:

| Observation | Dimensions |
| --- | ---: |
| Body-frame base linear velocity | 3 |
| Body-frame base angular velocity | 3 |
| Projected gravity | 3 |
| Velocity command `[vx, vz, yaw_rate]` | 3 |
| Relative joint positions / position error | 12 |
| Relative joint velocities | 12 |
| Previous action | 12 |
| **Total** | **48** |

Action and control interface:

| Parameter | Value |
| --- | --- |
| Policy action bounds | `[-100, 100]` |
| Action dimensions | 12 |
| Action scale | `0.25 rad` |
| Torque limits | URDF effort limits |

```text
q_target = q_home + 0.25 * action
```

The policy outputs target offsets consumed by the actuator model; it does not directly set simulated joint angles.

Home pose:

| Leg | Hip | Thigh | Calf |
| --- | ---: | ---: | ---: |
| FR | -0.05 | 0.75 | -1.30 |
| FL | +0.05 | 0.75 | -1.30 |
| RR | -0.05 | 0.85 | -1.30 |
| RL | +0.05 | 0.85 | -1.30 |

## Actuation and Control

The default actuator is `actuator_net`, using:

```text
resources/actuator_nets/unitree_go1.pt
```

Actuator-network inputs:

- `q - q_target` history
- joint-velocity history
- current and previous two policy-step samples

| Quantity | Value |
| --- | ---: |
| Policy timestep | 0.020 s |
| Policy rate | 50 Hz |
| Physics timestep | 0.005 s |
| Physics rate | 200 Hz |
| Substeps / policy step | 4 |
| Actuator history update | Once per policy step |
| Torque application | Chrono force motors |
| Torque clipping | URDF effort limits |

## Simulation Backends

|  | NSC | SCM |
| --- | --- | --- |
| Purpose | Flat rigid terrain | Deformable terrain |
| Selection | Default | `--env-backend scm` |
| Contact/system | NSC | `ChSystemSMC` |
| Friction | Fixed `0.8` | SCM soil model |
| Solver | - | `BARZILAIBORWEIN` |
| Solver iterations | - | `60` |

SCM soil parameters:

| SCM parameter | Value |
| --- | ---: |
| Bekker `Kphi` | `3e6` |
| Bekker `Kc` | `0` |
| Bekker `n` | `1.1` |
| Cohesion | `0` |
| Friction angle | `30 deg` |
| Janosi shear | `0` |
| Elastic stiffness | `2e9` |
| Damping | `3e4` |
| Gravity | `(0, -9.81, 0)` |

## Reward

```text
reward = 0.02 * raw_reward
```

Positive reward clipping is off. Terminal reward is zero.

| Term | Weight |
| --- | ---: |
| `tracking_lin_vel` | +1.5 |
| `tracking_ang_vel` | +0.75 |
| `lin_vel_z` | -2.0 |
| `ang_vel_xy` | -0.05 |
| `torques` | -0.0002 |
| `dof_acc` | -2.5e-7 |
| `flat_orientation_l2` | -2.5 |
| `feet_air_time` | +0.25 |
| `action_rate` | -0.01 |
| `termination` | 0.0 |

## Training Configuration

| Parameter | Value |
| --- | ---: |
| Environments | 24 |
| Steps / environment | 256 |
| Rollout size | 6144 |
| Batch size | 1536 |
| PPO epochs | 3 |
| Learning rate | `1e-4` |
| Final learning rate | `1e-4` |
| Clip range | `0.1` |
| Target KL | `0.015` |
| Entropy coefficient | `0.001` |
| Gamma | `0.99` |
| GAE lambda | `0.95` |
| Max gradient norm | `1.0` |
| Checkpoint frequency | 1,000,000 steps |

Default training uses 24 parallel environments and a rollout of 6,144 transitions per PPO update.

## Randomization

| Parameter | Configuration |
| --- | --- |
| Friction | Fixed `0.8` |
| Joint reset multiplier | `1.0` |
| Root X/Z offset | `U(-0.5, 0.5)` |
| Root yaw | `U(-pi, pi)` |
| Root linear velocity | Zero |
| Root angular velocity | Zero |
| Observation noise | Enabled |
| Added base mass | `U(-1.0, 3.0)` |
| COM offset | None |
| External pushes | None |

## Training / Fine-Tuning

New NSC training run:

```bash
python train_policy.py \
  --save-dir runs/default_new_run \
  --target-total-steps 100000000 \
  --checkpoint-freq 1000000
```

SCM fine-tuning run:

```bash
python train_policy.py \
  --env-backend scm \
  --save-dir runs/default_scm_finetune_repro \
  --resume-model runs/default_baseline/checkpoints/flat_150m_baseline.zip \
  --target-total-steps 570990864 \
  --max-steps 1000 \
  --actuator-model actuator_net \
  --num-envs 24 \
  --n-steps 256 \
  --batch-size 1536 \
  --n-epochs 3 \
  --learning-rate 0.00003 \
  --learning-rate-final 0.00003 \
  --clip-range 0.05 \
  --target-kl 0.001 \
  --ent-coef 0.001 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --max-grad-norm 0.5 \
  --checkpoint-freq 1000000
```

TensorBoard:

```bash
tensorboard --logdir runs/default_new_run/tensorboard --port 6006
```

## Evaluation and Diagnostics

Fixed-command NSC viewer:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0
```

NSC diagnostic:

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

SCM diagnostic:

```bash
python diagnose_policy.py \
  --env-backend scm \
  runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0 \
  --episodes 1 \
  --max-steps 1000 \
  --out diagnostics/default_scm_forward_eval \
  --log-every-step
```

`<!-- Optional diagnostic visualization -->`

## Documentation

- [docs/reproducibility.md](docs/reproducibility.md) — exact environment and run reproduction.
- [docs/reproduction_ladder.md](docs/reproduction_ladder.md) — flat baseline to SCM fine-tune lineage.
- [docs/training_roadmap.md](docs/training_roadmap.md) — training settings and TensorBoard metrics.
