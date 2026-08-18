# Reproduction Ladder

This ladder is the current default path for reproducing the active baseline behavior.

The baseline model zip is not tracked by git. Before running the baseline viewer or diagnostics from a fresh checkout, place the model at:

```text
runs/default_baseline/checkpoints/default_baseline.zip
```

## 1. Compile

```bash
python -m py_compile go1_env.py go1_scm_env.py train_stand.py diagnose_policy.py view_stand_policy.py view_scm_policy_vsg.py diagnostics.py ppo_compat.py project_config.py
```

## 2. Environment Smoke

```bash
python - <<'PY'
from go1_env import Go1Env
env = Go1Env()
obs, info = env.reset(seed=1)
print(obs.shape)
print(env.action_space)
print(info["actuator_model"])
print(info["material"]["ground_friction"])
print(info["command_mode"])
env.close()
PY
```

Expected:

```text
(48,)
Box(-100.0, 100.0, (12,), float32)
actuator_net
0.8
default
```

## 3. View Baseline

```bash
python view_stand_policy.py
```

## 4. Fixed Forward Diagnostic

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

Outputs:

```text
diagnostics/default_forward_eval/summary.json
diagnostics/default_forward_eval/episodes.json
diagnostics/default_forward_eval/timeline.csv
```

## 5. Fixed Lateral Diagnostic

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

## 6. Train A New Default Run

```bash
python train_stand.py \
  --save-dir runs/default_new_run \
  --target-total-steps 100000000 \
  --checkpoint-freq 1000000
```

## 7. SCM Deformable Terrain Diagnostic

CRM/SPH was tested, but it is too computationally heavy for practical fine-tuning on this setup. SCM is the active deformable-terrain backend.

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

## 8. SCM VSG Viewer

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

## 9. Continue A Run

Use the saved model and state from the run:

```bash
python train_stand.py \
  --save-dir runs/default_new_run_continued \
  --resume-model runs/default_new_run/final_model.zip \
  --resume-state runs/default_new_run/final_model.state.json \
  --target-total-steps 150000000 \
  --checkpoint-freq 1000000
```

`--resume-model` loads the policy. `--resume-state` restores the global timestep counter used for continuation accounting.
