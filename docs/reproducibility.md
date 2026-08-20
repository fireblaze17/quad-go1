# Reproducibility

This document contains the setup and command path needed to reproduce the project environment and run the baseline policies.

## Required Artifacts

Model zips are not tracked by git. Place these files before running the viewer, diagnostics, or fine-tuning commands:

```text
runs/default_baseline/checkpoints/flat_150m_baseline.zip
runs/default_baseline/checkpoints/default_baseline.zip
```

The promoted baseline is:

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

It contains the 571M-step SCM-fine-tuned policy.

## Source-Built Chrono Environment

The project environment is `chrono-src`. It is a conda environment plus a source-built Chrono/VSG install; packaged `pychrono` is not used.

Normal setup from a fresh checkout:

```bash
conda env create -f environment.yml
conda activate chrono-src
bash scripts/setup_chrono_source.sh

conda deactivate
conda activate chrono-src
```

The source-build script uses:

```text
Chrono source commit: 9faf13dd8f1128dd75ed233a9627027b0422c3f7
Chrono install:       ~/chrono_builds/chrono-install
VSG install:          ~/chrono_builds/vsg-install
Python env:           chrono-src
BUILD_JOBS default:   8
```

Required Chrono modules:

```text
Python
Parsers
Irrlicht
Vehicle
Vehicle models
VSG
FSI
FSI SPH
```

The setup script installs conda activation hooks under:

```text
$CONDA_PREFIX/etc/conda/activate.d/chrono_source.sh
$CONDA_PREFIX/etc/conda/deactivate.d/chrono_source.sh
```

Those hooks set the source-built Chrono library path, Python path, and VSG asset path.

## Source-Build Safeguards

The setup script handles the compatibility issues found while reproducing `chrono-src`:

- Builds Chrono from source instead of installing packaged `pychrono`.
- Builds VSG first and passes explicit `vsg_DIR`, `vsgXchange_DIR`, and `vsgImGui_DIR` paths to Chrono.
- Uses `pychrono.vsg3d` for VSG validation.
- Installs `libirrlicht-dev` on apt-based systems if the Irrlicht headers/library are missing.
- Pins the conda package builds that fixed the `pkg-config` / `xcb.pc` parsing issue.
- Passes explicit Irrlicht and Thrust paths to Chrono CMake.
- Uses CUDA 12.8 build packages for Chrono FSI/SPH while keeping PyTorch CPU-only.
- Defaults to `BUILD_JOBS=8` so WSL does not compile Chrono with every available core.

If the C++ compile is interrupted after it has already started, resume without deleting the partial build:

```bash
cd /home/ankus/Robot
source /home/ankus/miniforge3/etc/profile.d/conda.sh

CLEAN_BUILD=0 CLEAN_CHRONO_BUILD=0 BUILD_JOBS=8 \
  bash scripts/reproduce_chrono_source_env.sh chrono-src-test "$HOME/chrono_repro_test"
```

Use `BUILD_JOBS=4` on a memory-constrained machine.

## Clean-Room Environment Test

To test reproducibility without touching the working `chrono-src` build:

```bash
cd /home/ankus/Robot
source /home/ankus/miniforge3/etc/profile.d/conda.sh

bash scripts/reproduce_chrono_source_env.sh chrono-src-test "$HOME/chrono_repro_test"

conda deactivate
conda activate chrono-src-test
```

Validate the build:

```bash
python - <<'PY'
import pychrono
import pychrono.parsers
import pychrono.irrlicht
import pychrono.vsg3d
import pychrono.vehicle as veh

print("pychrono ok")
print("SCM terrain support:", hasattr(veh, "SCMTerrain"))
PY
```

Validate the project environments:

```bash
python - <<'PY'
from go1_env import Go1Env
from go1_scm_env import Go1SCMEnv

env = Go1Env(max_steps=1)
obs, info = env.reset(seed=1)
print("flat", obs.shape, info["env_backend"])
env.close()

env = Go1SCMEnv(max_steps=1)
obs, info = env.reset(seed=1)
print("scm", obs.shape, info["env_backend"])
env.close()
PY
```

## Static Check

```bash
python -m py_compile go1_env.py go1_scm_env.py train_stand.py diagnose_policy.py view_stand_policy.py view_scm_policy_vsg.py diagnostics.py ppo_compat.py project_config.py
python train_stand.py --help
python view_stand_policy.py --help
python view_scm_policy_vsg.py --help
python diagnose_policy.py --help
```

## Baseline Commands

View the promoted baseline on NSC:

```bash
python view_stand_policy.py
```

View the promoted baseline on SCM:

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

Run a fixed-command NSC diagnostic:

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

Run a fixed-command SCM diagnostic:

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

Open TensorBoard for a new run:

```bash
tensorboard --logdir runs/default_new_run/tensorboard --port 6006
```
