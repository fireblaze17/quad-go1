# Unitree Go1 MuJoCo RL Project

This project is a staged robotics ML project using the Unitree Go1 quadruped in
MuJoCo. The long-term goal is to build from basic simulation toward locomotion,
rollout data collection, a learned world model, and eventually hierarchical
control.

Current milestone:

```text
Robust standing policy trained and evaluated successfully.
```

## Setup

Environment:

```text
conda env: quad-go1
```

Main libraries:

```text
mujoco
gymnasium
numpy
stable-baselines3
```

Robot model:

```text
c:\Learning code\mujoco_menagerie\unitree_go1\scene.xml
```

`scene.xml` is used because it includes the Go1 model plus the ground/world
scene.

## Current Files

```text
go1_env.py
```

Gymnasium environment wrapping the Go1 MuJoCo simulation.

```text
train_stand.py
```

Trains the PPO standing policy.

```text
view_trained_policy.py
```

Loads the saved policy and shows it in the MuJoCo viewer.

```text
evaluate_stand.py
```

Runs deterministic evaluation episodes and reports survival/reward statistics.

```text
test_load_go1.py
```

Early model-loading and inspection script.

```text
calculate_fall_height.py
```

Calculates the fall-height threshold from the official Go1 home keyframe.

## What Was Built

The Go1 model exposes:

```text
nq = 19 position values
nv = 18 velocity values
nu = 12 actuator controls
```

The observation is:

```text
qpos concatenated with qvel
```

The action is:

```text
12 motor command offsets
```

Actions are not used as raw joint targets. Instead, the policy outputs small
offsets around the official home pose:

```python
ctrl = default_ctrl + action_scale * action
```

This made training much less explosive than allowing full-range motor commands
immediately.

## Main Challenges

The first simple reward only encouraged the robot to keep its body above a fall
height. The robot found loopholes: flopping, leg shuffling, sliding, and slow
spinning.

The standing reward was shaped step by step to close those loopholes:

```text
alive reward              -> do not fall
upright reward            -> keep the body vertical
motor control penalty     -> avoid wild commands
home-pose penalty         -> keep legs near standing pose
base velocity penalty     -> avoid sliding
angular velocity penalty  -> avoid spinning
base position penalty     -> stay near the starting spot
```

The environment also resets to the official MuJoCo Menagerie `home` keyframe and
adds reset noise so the policy is not trained on only one exact starting pose.

## Standing Policies

First successful standing policy:

```text
go1_stand_policy_v1.zip
```

Current main robust standing policy:

```text
go1_stand_policy_v2.zip
go1_stand_policy.zip
```

`go1_stand_policy.zip` is currently a copy of `v2`.

## Current Result

Robust standing policy settings:

```text
reset joint noise: 0.10
reset velocity noise: 0.10
training timesteps: 100,000
```

Evaluation:

```text
fall rate: 0.0%
```

The policy survived the full evaluation episodes under reset noise and is the
current standing baseline.

## Next Milestones

1. Record standing policy videos.
2. Train a forward walking policy.
3. Train turning or backward-walking policies.
4. Collect rollout data from multiple behaviors.
5. Train a world model on `(obs, action, next_obs)` transitions.
6. Use the world model later for planning or hierarchical control experiments.

