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
Status: first Chrono physics demo working
```

## Milestone 1: Basic Chrono Physics Demo

The first Chrono demo is implemented in:

```text
chrono_demo.py
```

It creates:

```text
Chrono physics system
Y-up gravity
fixed ground body
dynamic falling box
Bullet collision system
Irrlicht visualization window
```

The box falls onto the ground, contacts are detected, and the box settles at the
expected height.

Important Chrono setup details:

```text
ChSystemNSC
SetGravityY()
SetCollisionSystemType(Type_BULLET)
ChContactMaterialNSC
AddBody(...)
DoStepDynamics(...)
ChVisualSystemIrrlicht
```

This milestone proves that basic Chrono physics, contact, and visualization are
working before adding robots or learning code.

## Next Steps

1. Clean up the Chrono demo code.
2. Add a more interesting terrain demo.
3. Start building/importing the quadruped model.
4. Wrap the Chrono simulation in a Gymnasium environment.
