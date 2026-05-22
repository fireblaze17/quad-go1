# Reproducibility

This file is the shortest path for another researcher to recreate the accepted
flat-ground standing baseline without reading the full decision history.

## Platform Assumptions

```text
OS tested:        WSL Ubuntu on Windows
Python:           3.12
PyChrono:         10.0.0
Gymnasium:        1.2.3
Stable-Baselines3 2.8.0
Torch:            2.11.0
World frame:      Chrono Y-up
Robot model:      models/go1/go1_chrono.urdf
```

The active development path is WSL Ubuntu with the `chrono-go1` conda
environment. Activate it before running project commands:

```bash
conda activate chrono-go1
```

After activation, use `python`. On Ankus's WSL machine, the equivalent explicit
interpreter is `/home/ankus/miniforge3/envs/chrono-go1/bin/python`.

The current setup uses Irrlicht through WSLg for visualization. If GUI windows
or the viewer start behaving strangely, close open GUI windows first. If that
does not clear it, run this from Windows PowerShell and then reopen the WSL
terminal:

```powershell
wsl --shutdown
```

If Irrlicht cannot use the default video driver, the viewer may print a fallback
message and continue with OpenGL.

## Why WSL Is The Primary Platform

Native Windows PyChrono was abandoned for this project after Windows Smart App
Control blocked unsigned Chrono extension binaries during import. The observed
failure mode included blocked files such as:

```text
Chrono_vehicle.dll
_parsers.pyd
```

When this happens, PyChrono modules may fail to import even though the conda
environment appears to be installed correctly. Because these are binary
extension loads, reinstalling Python packages is usually not the useful first
move.

Recommended response:

1. Switch to WSL Ubuntu for this repo.
2. Create or use the WSL `chrono-go1` conda environment.
3. Activate `chrono-go1` and run commands with `python`.
4. Treat the old native Windows repo as backup only.

Alternative Windows-only recovery is to open Windows Security, check Protection
History, and allow the blocked Chrono binaries or change the local application
control policy. That path is not the project baseline because it is machine
policy dependent and easy to break again.

## Environment Setup

Create the conda environment from the root file:

```bash
conda env create -f environment.yml
```

Activate it if conda is initialized in your shell:

```bash
conda activate chrono-go1
```

The commands below assume `conda activate chrono-go1` has already run.

Quick syntax check:

```bash
python -m py_compile go1_env.py diagnostics.py diagnose_policy.py view_env.py view_stand_policy.py evaluate_stand.py train_stand.py friction_curriculum.py run_regression.py project_config.py
```

## Stable-Baselines3 Device Choice

`train_stand.py`, `evaluate_stand.py`, and `view_stand_policy.py` force
Stable-Baselines3 to load/run PPO on CPU:

```python
device="cpu"
```

Reason: this project uses an MLP PPO policy with CPU-bound Chrono rollouts. SB3
can auto-select CUDA when it is available, but it warns that PPO without a CNN
policy is usually intended for CPU. In this project that warning matched lower
training FPS, so CPU is the reproducible default.

Tradeoff: GPU acceleration is left unused for this policy, but training avoids
small CPU/GPU transfer overheads and the warning noise. Revisit only if the
policy architecture changes substantially, for example to image observations or
a CNN.

`project_config.py` stores shared project paths and runtime defaults, including
the fixed-friction baseline, the accepted friction A baseline, the current
B-capable AB baseline, the default viewer friction range, and the SB3 device.

## Checkpoint Availability

`runs/` is intentionally gitignored. Accepted checkpoints are local artifacts
and are not stored in GitHub because model files are much larger than the code
and docs. A fresh clone can run the environment, train from scratch, and produce
new checkpoints. Evaluating the accepted results requires copying the accepted
checkpoint folders into `runs/` out-of-band.

Required checkpoints for reproducing the accepted standing results:

```text
runs/stand_base_v2/final_model.zip          fixed-friction standing v2
runs/stand_friction_a_07_09/final_model.zip friction A accepted checkpoint
runs/stand_friction_ab_065_095/final_model.zip current/default B-capable checkpoint
```

Keep each checkpoint's `args.json` and `env_constants.json` beside
`final_model.zip` when sharing or archiving a run.

## Baseline Commands

Zero-action environment check:

```bash
python view_env.py
```

Train the accepted standing task:

```bash
python train_stand.py --terrain flat --friction-min 0.8 --friction-max 0.8 --timesteps 500000 --seed 1 --save-dir runs/stand
```

`train_stand.py` saves checkpoints every `25_000` steps by default. Override
with `--checkpoint-freq`; use `--checkpoint-freq 0` to disable intermediate
checkpoints while still saving `final_model.zip`.

Evaluate:

```bash
python evaluate_stand.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8 --episodes 10
```

View:

```bash
python view_stand_policy.py runs/stand/final_model.zip --terrain flat --friction-min 0.8 --friction-max 0.8
```

Curriculum sanity checks:

```bash
python friction_curriculum.py status
python friction_curriculum.py all-commands
```

## Current Viewer Default

`view_stand_policy.py` now defaults to the accepted B-capable AB checkpoint and
the full B range, so pressing Run in VS Code opens:

```text
policy:   runs/stand_friction_ab_065_095/final_model.zip
terrain:  flat
friction: 0.6-1.0
```

The default console output is compact and focused on the current acceptance
signals:

```text
h up xz act dact jvel tilt foot_min load_imb slip vfoot nonfoot_max
```

Use `--full-diagnostics` when debugging per-foot heights, foot load ranges, or
calf/thigh/hip contact loads in detail.

## Current B-Capable Commands

The current/default accepted standing policy is the AB checkpoint, trained on
`0.65-0.95` and accepted on the full B range `0.6-1.0`.

Evaluate:

```bash
python evaluate_stand.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0 --episodes 30
```

View:

```bash
python view_stand_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0
```

Headless tilt/contact diagnosis:

```bash
python diagnose_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.6 --friction-max 1.0 --episodes 30 --out diagnostics/ab_on_b_range
```

Eval + diagnosis regression wrapper:

```bash
python run_regression.py runs/stand_friction_ab_065_095/final_model.zip --name ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100
```

The wrapper writes `evaluate_stdout.txt`, `diagnose_stdout.txt`,
`summary.json`, `episodes.json`, and `regression_summary.json` under
`diagnostics/<name>/`. Use it before accepting any future standing challenger,
then verify the viewer manually.

Expected B-capable signature:

```text
survival_rate:       1.000
mean_length:         1000.0
min_upright_score:   about 0.999
foot_contact_error:  about 0.010
min_foot_load:       about 23 N
nonfoot_load_max:    0
termination_reasons: {'truncated': 30}
```

## Current C-Range Comparison

The AB checkpoint is also the current C-capable reference. It was trained on
`0.65-0.95`, accepted for B on `0.6-1.0`, and then tested directly on the wider
C range `0.5-1.1`.

AB-on-C reference commands:

```bash
python run_regression.py runs/stand_friction_ab_065_095/final_model.zip --name ab_on_c_range --friction-min 0.5 --friction-max 1.1 --episodes 100
python view_stand_policy.py runs/stand_friction_ab_065_095/final_model.zip --terrain flat --friction-min 0.5 --friction-max 1.1
```

Expected AB-on-C signature:

```text
survival_rate:       1.000
mean_length:         1000.0
mean_reward:         about 1138.8
min_upright_score:   about 0.999
foot_contact_error:  about 0.0094
min_foot_load:       about 23.4 N
mean_abs_xz_vel:     about 0.012
max_abs_xz_vel:      about 0.020
diagnosis:           no_tilt_threshold_crossing in 100/100 episodes
nonfoot load:        0
```

Final C comparison:

| Model | Timesteps | Survival | Upright | Contact error | Min foot load | X/Z drift | Diagnosis | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AB-on-C | 700k staged | 1.000 | 0.999 | 0.009350 | 23.431 N | 0.011950 | no tilt crossing | accepted |
| C-from-AB | 1.0M staged | 1.000 | 0.999 | 0.016239 | 20.134 N | 0.045557 | FL-heavy left/right bias | rejected |
| Scratch C seed1 | 700k | 1.000 | 0.996 | 0.016500 | 19.784 N | 0.024861 | foot unload before tilt | rejected |
| Scratch C seed2 | 700k | 1.000 | 0.987 | 0.104047 | 15.162 N | 0.044395 | foot unload before tilt | rejected |
| Scratch C seed3 | 700k | 1.000 | 0.974 | 0.249241 | 5.966 N | 0.039411 | foot unload before tilt | rejected |

This is why reproducibility checks compare challengers against AB-on-C instead
of assuming the newest, widest-range, or equal-budget scratch model is better.
Survival is required, but it does not certify clean standing.

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
