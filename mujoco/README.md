# Unitree Go1 MuJoCo Reference

This folder is the old MuJoCo baseline. The active project has moved to Project
Chrono, but these files stay as reference for reward shaping, Menagerie model
values, and the first successful standing-policy workflow.

Historical milestone:

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

Gymnasium environment wrapping the Go1 MuJoCo simulation. It now supports:

```python
Go1Env(task="stand")
Go1Env(task="walk")
```

```text
train_stand.py
```

Trains the PPO standing policy. Generated `.zip` policy artifacts are ignored
and are not kept in git.

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

## Walking Task Setup

The environment now has a separate walking mode:

```python
Go1Env(task="walk")
```

The walking reward is intentionally separate from the standing reward.

Standing rewards include terms that discourage movement:

```text
base position penalty
base velocity penalty
home-pose penalty
```

Those are useful for standing, but they would fight against walking. The first
walking reward instead emphasizes:

```text
alive reward
upright reward
forward velocity reward
motor control penalty
small pose penalty
angular velocity penalty
```

The old walking placeholder scripts were removed during cleanup. New walking
work should happen in the Chrono env unless this MuJoCo baseline is deliberately
revived for comparison.

## Standing Policies

The first successful standing policy artifacts were removed from git during
cleanup. Keep regenerated `.zip` files local or store them through an artifact
system, not in the repo.

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

## Videos

Generated videos were removed from git during cleanup. Keep rollout videos local
or publish them outside the source tree when they are needed for demos.

## Next Milestones

1. Train a forward walking policy.
2. Train turning or backward-walking policies.
3. Collect rollout data from multiple behaviors.
4. Train a world model on `(obs, action, next_obs)` transitions.
5. Use the world model later for planning or hierarchical control experiments.
