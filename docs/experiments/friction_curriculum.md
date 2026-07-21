# Flat Friction Curriculum Archive

> Historical archive: this document records the old v2 A/AB/C friction
> curriculum at a high level. The helper scripts that used to drive this flow
> were removed during the 48D / 50 Hz cleanup. Do not use this page as an
> active reproduction guide.

## Summary

The old v2 standing lineage explored increasingly wide flat-ground friction
ranges:

```text
runs/stand_base_v2
-> runs/stand_friction_a_07_09
-> runs/stand_friction_ab_065_095
-> runs/stand_friction_c_05_11
```

These runs are kept as historical evidence for how the project evolved. They
are not current v3.1 claims.

## Retrospective

The friction curriculum was useful for learning how to organize fixed-range and
randomized-range experiments. Later work changed the interpretation: for a
standing-only policy with no pushes or horizontal disturbances, fixed-friction
slices are not a meaningful robustness pass. Friction should be revisited after
random push recovery creates real horizontal shear demands.

## Current Status

```text
active environment: 48D relative-state v3.1 plus zero command
active timing: 200 Hz physics / 50 Hz control
active evaluator: diagnose_policy.py
active friction status: not accepted
```

For current commands, use:

```text
docs/reproducibility.md
docs/reproduction_ladder.md
docs/training_roadmap.md
```
