# Reproducibility

## Baseline Artifact

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

This is the neutral checkpoint copy used as the active baseline. The original training artifact remains in `runs/` as a backup, but active docs and code use this path.
It currently contains the 571M-step SCM-fine-tuned policy.

Run artifacts and model zips are ignored by git. On a fresh checkout, copy or download the baseline model to this exact path before running the baseline viewer or diagnostics.

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

The script also installs activation hooks under the active conda env:

```text
$CONDA_PREFIX/etc/conda/activate.d/chrono_source.sh
$CONDA_PREFIX/etc/conda/deactivate.d/chrono_source.sh
```

Those hooks set:

```text
LD_LIBRARY_PATH = ~/chrono_builds/chrono-install/lib:~/chrono_builds/vsg-install/lib:/usr/lib/wsl/lib:$CONDA_PREFIX/lib:...
PYTHONPATH      = ~/chrono_builds/chrono-install/share/chrono/python:...
VSG_FILE_PATH   = ~/chrono_builds/vsg-install/share/vsgExamples
```

## Source-Build Issues Already Accounted For

These are the concrete build/runtime problems encountered while creating the working `chrono-src` setup and how the current setup files handle them.

- Packaged `pychrono` was not enough for this project because VSG import hit an ImGui/VSG ABI mismatch:
  ```text
  libChrono_vsg.so: undefined symbol: ImGui::Image(...)
  ```
  The environment file intentionally does not install packaged `pychrono`; `scripts/setup_chrono_source.sh` builds Chrono from source instead.

- The VSG Python module name is `pychrono.vsg3d`, not `pychrono.vsg`. The setup validation imports `pychrono.vsg3d`.

- Source-built PyChrono modules require the matching shared libraries. Importing the generated Python package without the matching `LD_LIBRARY_PATH` can load the wrong Chrono libraries and produce undefined-symbol errors. The setup script installs activation hooks so the source-built library and Python paths are active after `conda activate`.

- System Irrlicht headers/libraries are required for the current flat viewer path. The script checks for:
  ```text
  /usr/include/irrlicht/irrlicht.h
  /usr/lib/x86_64-linux-gnu/libIrrlicht.so
  ```
  If missing on an apt-based system, `scripts/setup_chrono_source.sh` installs `libirrlicht-dev` before configuring Chrono.

- VSG configuration failed in a clean-room test when conda solved to a different `pkg-config` build that could not parse `xcb.pc`:
  ```text
  Unknown keyword 'Libs.private' in xcb.pc
  The following required packages were not found: xcb
  ```
  The environment file pins the working package builds, and `scripts/setup_chrono_source.sh` force-installs these same builds before configuring VSG/Chrono:
  ```text
  git 2.55.0 pl5321h5685339_1
  cmake 4.4.1 hc85cc9f_0
  ninja 1.13.2 h171cf75_0
  pkg-config 0.29.2 h4bc722e_1009
  xorg-libxau 1.0.12 hb03c661_1
  xorg-libxdmcp 1.1.5 hb03c661_1
  xorg-xproto 7.0.31 hb9d3cd8_1008
  ```

- Chrono's VSG module needs installed VSG CMake configs. The setup script builds VSG first, checks for:
  ```text
  vsgConfig.cmake
  vsgXchangeConfig.cmake
  vsgImGuiConfig.cmake
  ```
  and passes the explicit `vsg_DIR`, `vsgXchange_DIR`, and `vsgImGui_DIR` paths to Chrono's CMake configure step.

- The working `chrono-src` CMake cache found Irrlicht and Thrust through these exact paths:
  ```text
  Irrlicht_INCLUDE_DIR = /usr/include/irrlicht
  Irrlicht_LIBRARY     = /usr/lib/x86_64-linux-gnu/libIrrlicht.so
  THRUST_INCLUDE_DIR   = $CONDA_PREFIX/targets/x86_64-linux/include
  ```
  The setup script passes those values explicitly so Chrono's Python wrappers, Irrlicht module, vehicle wrappers, and FSI/SPH targets configure the same way in a clean-room build.

- The upstream VSG helper also builds `vsgExamples` and may append to shell startup files. The project setup trims the helper after the VSG libraries needed by Chrono are installed, because the examples stage is not needed for this project.

- Chrono FSI/SPH requires CUDA build components. The environment uses CUDA 12.8 packages for building Chrono FSI/SPH while keeping PyTorch CPU-only:
  ```text
  torch==2.13.0+cpu
  ```
  This avoids mixing the project runtime with unrelated CUDA Python wheels.

## Clean-Room Environment Test

To test reproducibility without touching the working `chrono-src` build, run the wrapper with a separate env name and isolated build root:

```bash
cd /home/ankus/Robot
source /home/ankus/miniforge3/etc/profile.d/conda.sh

bash scripts/reproduce_chrono_source_env.sh chrono-src-test "$HOME/chrono_repro_test"

conda deactivate
conda activate chrono-src-test
```

The wrapper creates or updates the requested conda env, refuses to write into the working `~/chrono_builds` root, sets all isolated Chrono/VSG build paths, and then calls `scripts/setup_chrono_source.sh`. The source build defaults to `BUILD_JOBS=8`, matching the successful project build and avoiding the WSL memory pressure caused by compiling with all available cores.

If the C++ compile is interrupted after it has already started, resume without deleting the partial build:

```bash
cd /home/ankus/Robot
source /home/ankus/miniforge3/etc/profile.d/conda.sh

CLEAN_BUILD=0 CLEAN_CHRONO_BUILD=0 BUILD_JOBS=8 \
  bash scripts/reproduce_chrono_source_env.sh chrono-src-test "$HOME/chrono_repro_test"
```

On a memory-constrained machine, use `BUILD_JOBS=4` for the resume command.

Validate the build:

```bash
python - <<'PY'
import pychrono
import pychrono.parsers
import pychrono.irrlicht
import pychrono.vsg3d
import pychrono.vehicle as veh
import pychrono.fsi as fsi

print("pychrono ok")
print("CRMTerrain:", hasattr(veh, "CRMTerrain"))
print("SPH VSG:", hasattr(fsi, "ChSphVisualizationVSG"))
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

## Environment Defaults

```text
actuator_model: actuator_net
alternate actuator: torque_limited_pd
contact: flat rigid ground
friction: 0.8
policy_dt: 0.02
physics_dt: 0.005
physics_substeps: 4
max_steps: 1000
action_clip: 100
action_scale: 0.25
reward_clipping: off
```

## Static Check

```bash
python -m py_compile go1_env.py go1_scm_env.py train_stand.py diagnose_policy.py view_stand_policy.py view_scm_policy_vsg.py diagnostics.py ppo_compat.py project_config.py
python train_stand.py --help
python view_stand_policy.py --help
python diagnose_policy.py --help
```

## Viewer

```bash
python view_stand_policy.py
```

Fixed command viewer:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx -0.5 \
  --fixed-command-vz 0.0 \
  --fixed-command-yaw-rate 0.0
```

Collision-shape viewer:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --visual-mesh-format none \
  --show-collision-boxes
```

## Diagnostics

Forward:

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

Lateral:

```bash
python diagnose_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --fixed-command-vx 0.0 \
  --fixed-command-vz 0.5 \
  --fixed-command-yaw-rate 0.0 \
  --episodes 1 \
  --max-steps 1000 \
  --out diagnostics/default_lateral_eval \
  --log-every-step
```

## Training

```bash
python train_stand.py \
  --save-dir runs/default_new_run \
  --target-total-steps 100000000 \
  --checkpoint-freq 1000000
```

TensorBoard:

```bash
tensorboard --logdir runs/default_new_run/tensorboard --port 6006
```

## SCM Deformable Terrain

CRM/SPH was tested, but it is too computationally heavy for practical fine-tuning on this setup. SCM is the active deformable-terrain backend.

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

SCM VSG viewer:

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
