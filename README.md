# Quad Go1 Chrono

Project Chrono robotics simulation project for a Unitree Go1-style quadruped.

The goal is to build a robotics ML stack using the same simulator family used by
the target lab:

1. Set up a basic PyChrono simulation.
2. Build or import a quadruped model.
3. Wrap the Chrono simulator as a Gymnasium environment.
4. Train standing and locomotion policies.
5. Collect rollout data from learned skills.
6. Train a world model.
7. Add hierarchical skill selection later.

## Why Chrono

This project was restarted around Project Chrono because the target research lab
uses Chrono as part of its open-source physics simulation stack.

The immediate goal is not to build the full learning system at once. The first
milestone is to get a simple Chrono simulation running and then rebuild the
robotics environment step by step.

## Previous Prototype

A MuJoCo prototype was built first to learn the core robotics RL pipeline:

```text
robot simulation -> Gymnasium environment -> PPO standing policy -> evaluation
```

That work is preserved in:

```text
mujoco/
```

The MuJoCo prototype includes a trained standing policy, evaluation scripts,
saved checkpoints, and before/after videos. It is kept as a reference, but the
main project direction from here is Project Chrono.

## Current Status

```text
Status: restarting foundation in Project Chrono
```

Next steps:

1. Create a Chrono/PyChrono environment.
2. Run a minimal falling-body or ground-contact simulation.
3. Add visualization.
4. Start building/importing the quadruped model.

