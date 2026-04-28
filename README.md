# Quad Go1

Robotics simulation and machine learning project for a Unitree Go1-style
quadruped using Project Chrono.

The goal is to build a clean robotics ML stack:

1. Set up a basic PyChrono simulation.
2. Build or import a quadruped model.
3. Wrap the Chrono simulator as a Gymnasium environment.
4. Train standing and locomotion policies.
5. Collect rollout data from learned skills.
6. Train a world model.
7. Add hierarchical skill selection later.

## Current Status

```text
Status: building the Project Chrono foundation
```

Next steps:

1. Create a Chrono/PyChrono environment.
2. Run a minimal falling-body or ground-contact simulation.
3. Add visualization.
4. Start building/importing the quadruped model.
