# Standing-Only Debug History

This note is historical. It records old standing-only work so it does not clutter the active locomotion documentation.

## What Was Tried

- Metric-gated standing curricula.
- Quiet-stand reward shaping.
- Friction and reset-noise sweeps aimed only at standing.
- Pose penalties, load-balance penalties, collision penalties, and action penalties for standing.
- Ideal position control, explicit torque-limited PD, and learned actuator tests.
- Zero-action home-pose sweeps and spawn-height experiments.
- Short runs that looked failed before stochastic training behavior had enough time to settle.
- Contact and solver experiments that were aimed at making a stationary pose survive.

## What Was Learned

- Standing-only success did not guarantee locomotion.
- Several standing policies found static or seated local minima.
- Collision geometry and solver behavior had to be fixed before reward conclusions were meaningful.
- Zero-action standing remained useful as a diagnostic, but it was not sufficient as the main training target.
- The current baseline moved away from standing-only training and uses full command locomotion as the active task.
- Friction-only standing tests were not strong evidence of locomotion robustness because standing does not demand the same horizontal shear as commanded walking and turning.
- Reset-noise work aimed only at standing is not part of the current training recipe.

## Current Status

The active baseline is documented in:

```text
docs/documentation.md
```

Use that document for current commands, reward terms, stack values, and reproduction steps.
