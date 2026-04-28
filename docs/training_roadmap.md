# Training Roadmap

This is the training plan for the Chrono Go1 environment.

## Current Blockers

Before training:

```text
add fall termination
add standing reward
create Chrono train_stand.py
```

`go1_env.py` still uses a placeholder reward. Training with `reward = 0.0`
would teach nothing.

## Stage 1: Standing On Flat Ground

Environment:

```python
Go1Env(terrain="flat", enable_motors=True)
```

Goal:

```text
learn stable standing with position-control offsets
```

Reward terms to add:

```text
alive reward
trunk height reward / fall penalty
upright orientation reward
joint pose penalty around home pose
base linear velocity penalty
base angular velocity penalty
action penalty
```

Termination:

```text
terminated = trunk_y < _TERM_HEIGHT
```

## Stage 2: Walking On Flat Ground

After standing is stable, train forward walking on rigid flat terrain.

Reward should differ from standing:

```text
alive reward
upright reward
forward velocity reward
action penalty
angular velocity penalty
small pose regularization
```

Standing penalties that prevent movement should be reduced or removed:

```text
base position penalty
strong base velocity penalty
strong home-pose penalty
```

## Stage 3: SCM Fine-Tuning

Once flat walking works, fine-tune on Chrono SCM deformable terrain:

```python
Go1Env(terrain="scm")
```

Reason:

```text
SCM is slower and harder, but it is the physics feature that motivates
using Chrono instead of only MuJoCo.
```

Tradeoff:

```text
flat first gives speed and stability
SCM later gives terrain realism
```

## Stage 4: Rollout Collection

Collect transition data from learned skills:

```text
obs
action
reward
next_obs
terminated/truncated
skill label
terrain type
```

## Stage 5: World Model

Train a model over transitions:

```text
(obs, action) -> next_obs
```

Initial target: short-horizon prediction.

Later target: planning or skill-level abstraction.

## Stage 6: Hierarchical Skill Selection

Use learned policies as low-level skills:

```text
stand
walk forward
turn
recover
terrain-specific variants
```

Higher-level controller chooses skills based on state and objective.
