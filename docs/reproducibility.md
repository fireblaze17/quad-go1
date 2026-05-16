# Reproducibility

This file is the shortest path for another researcher to recreate the accepted
flat-ground standing baseline without reading the full decision history.

## Platform Assumptions

```text
OS tested:        Windows
Python:           3.12
PyChrono:         10.0.0
Gymnasium:        1.2.3
Stable-Baselines3 2.8.0
Torch:            2.11.0
World frame:      Chrono Y-up
Robot model:      models/go1/go1_chrono.urdf
```

The current setup uses Irrlicht for visualization. If Irrlicht cannot use the
default video driver, the viewer may print a fallback message and continue with
OpenGL.

## Environment Setup

Create the conda environment from the root file:

```powershell
C:\Users\ankus\anaconda3\Scripts\conda.exe env create -f environment.yml
```

Activate it if conda is initialized in your shell:

```powershell
conda activate chrono-go1
```

The commands below use the environment's `python.exe` directly, so they also
work when shell activation is unavailable.

## Baseline Commands

Zero-action environment check:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe view_env.py
```

Train the accepted standing task:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000 --seed 1 --save-dir runs/stand
```

Evaluate:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe evaluate_stand.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10
```

View:

```powershell
C:\Users\ankus\anaconda3\envs\chrono-go1\python.exe view_stand_policy.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8
```

## Expected Standing V2 Metrics

The accepted baseline evaluation should be in this neighborhood:

```text
survival_rate:       1.000
mean_length:         1000.0
min_trunk_y:         about 0.337
min_upright_score:   about 1.000
mean_abs_xz_vel:     about 0.007
mean_abs_action:     about 0.304
mean_foot_load:      about 32 N
termination_reasons: {'truncated': 10}
```

The policy viewer should show:

```text
calf_load/thigh_load/hip_load = 0 on all legs
nonfoot_load_max = 0 on all legs
all four feet near the same height after settling
no foot permanently at zero load
no obvious visual vibration
```

## Run Metadata

`train_stand.py` writes reproducibility metadata into the save directory:

```text
args.json
env_constants.json
```

Keep these files with any saved policy checkpoint. They record the command-line
configuration and the environment constants that define the training task.

## Known Non-Determinism

RL training is not bit-for-bit deterministic here. Sources include:

- PPO sampling and neural network initialization
- Chrono contact solver iteration details
- floating-point differences across hardware and package builds
- visualization not being part of evaluation

For scientific comparison, compare multiple seeds and report the exact command,
environment file, reward constants, collision whitelist, and final evaluation
output.
