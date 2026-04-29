"""Evaluate a trained Chrono Go1 standing policy without rendering."""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from go1_env import Go1Env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy",
        type=Path,
        nargs="?",
        default=Path("runs/stand/final_model.zip"),
        help="Path to a Stable-Baselines3 policy zip.",
    )
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.8)
    parser.add_argument("--friction-max", type=float, default=0.8)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.policy.exists():
        raise FileNotFoundError(
            f"Policy not found: {args.policy}. Pass the model zip path, for "
            "example: evaluate_stand.py runs/stand/final_model.zip"
        )

    env = Go1Env(
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(args.friction_min, args.friction_max),
    )
    model = PPO.load(args.policy)

    episode_rewards = []
    episode_lengths = []
    failures = 0
    frictions = []
    min_heights = []
    min_upright_scores = []
    min_axis_scores = {"trunk_x_up": [], "trunk_y_up": [], "trunk_z_up": []}
    mean_abs_actions = []
    max_abs_actions = []
    termination_reasons = {}

    for _ in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        frictions.append(info["ground_friction"])
        min_height = float("inf")
        min_upright_score = float("inf")
        episode_min_axes = {"trunk_x_up": float("inf"), "trunk_y_up": float("inf"), "trunk_z_up": float("inf")}
        termination_reason = None
        action_abs_sum = 0.0
        action_abs_max = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            abs_action = np.abs(action)
            action_abs_sum += float(np.mean(abs_action))
            action_abs_max = max(action_abs_max, float(np.max(abs_action)))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            reward_terms = info.get("reward_terms", {})
            min_height = min(min_height, reward_terms.get("trunk_y", float("inf")))
            min_upright_score = min(
                min_upright_score,
                reward_terms.get("upright_score", float("inf")),
            )
            for key in episode_min_axes:
                episode_min_axes[key] = min(
                    episode_min_axes[key],
                    reward_terms.get(key, float("inf")),
                )
            termination_reason = info.get("termination_reason")

        failures += int(terminated)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        min_heights.append(min_height)
        min_upright_scores.append(min_upright_score)
        for key, value in episode_min_axes.items():
            min_axis_scores[key].append(value)
        mean_abs_actions.append(action_abs_sum / max(1, steps))
        max_abs_actions.append(action_abs_max)
        key = termination_reason or ("truncated" if truncated else "unknown")
        termination_reasons[key] = termination_reasons.get(key, 0) + 1

    env.close()

    print(f"episodes: {args.episodes}")
    print(f"survival_rate: {(1.0 - failures / args.episodes):.3f}")
    print(f"mean_reward: {np.mean(episode_rewards):.3f}")
    print(f"mean_length: {np.mean(episode_lengths):.1f}")
    print(f"mean_abs_action: {np.mean(mean_abs_actions):.3f}")
    print(f"max_abs_action: {np.max(max_abs_actions):.3f}")
    print(f"min_trunk_y: {np.min(min_heights):.3f}")
    print(f"min_upright_score: {np.min(min_upright_scores):.3f}")
    print(
        "min_axis_up:",
        {
            key: round(float(np.min(values)), 3)
            for key, values in min_axis_scores.items()
        },
    )
    print(f"termination_reasons: {termination_reasons}")
    if any(friction is not None for friction in frictions):
        print(f"friction_min_seen: {np.nanmin(frictions):.3f}")
        print(f"friction_max_seen: {np.nanmax(frictions):.3f}")


if __name__ == "__main__":
    main()
