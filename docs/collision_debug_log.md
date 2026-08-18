# Collision Diagnostics

## Current Use

Collision diagnostics are retained for viewer and headless evaluation. They are not a separate training stack.

The viewer can show collision shapes only:

```bash
python view_stand_policy.py runs/default_baseline/checkpoints/default_baseline.zip \
  --visual-mesh-format none \
  --show-collision-boxes
```

## Active Collision Setup

- Trunk and feet are physical contact bodies.
- Leg contact shapes are available for diagnostics and visualization.
- Self-collision filtering is kept to prevent neighboring robot links from fighting each other.
- The active reward does not penalize thigh/calf contact directly.
- Termination is trunk contact over the configured force limit.

## What To Inspect

- `foot_load_*`
- `calf_load_*`
- `thigh_load_*`
- `foot_friction_usage_*`
- `foot_load_share_*`
- `min_foot_load`
- `foot_load_imbalance`
- `max_joint_frame_separation`

These fields appear in viewer diagnostic prints and in `timeline.csv` from `diagnose_policy.py --log-every-step`.

Collision diagnostics should be checked whenever a policy appears to improve through sitting, dragging, scuttling, or supporting itself on non-foot geometry. The current default reward does not directly penalize thigh/calf contact, so the diagnostics are the guardrail for catching that behavior.
