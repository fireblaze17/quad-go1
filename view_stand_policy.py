"""View a trained Chrono Go1 standing policy in Irrlicht."""

import argparse
from pathlib import Path

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
    parser.add_argument("--max-steps", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.policy.exists():
        raise FileNotFoundError(
            f"Policy not found: {args.policy}. Pass the model zip path, for "
            "example: view_stand_policy.py runs/stand/final_model.zip"
        )

    env = Go1Env(
        render_mode="human",
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(args.friction_min, args.friction_max),
    )
    model = PPO.load(args.policy)
    obs, _ = env.reset()
    step = 0

    try:
        while env.render():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if step % 100 == 0:
                reward_terms = info.get("reward_terms", {})
                print(
                    "step:",
                    step,
                    "mean_abs_action:",
                    round(float(abs(action).mean()), 3),
                    "max_abs_action:",
                    round(float(abs(action).max()), 3),
                    "trunk_y:",
                    round(float(reward_terms.get("trunk_y", 0.0)), 3),
                    "upright:",
                    round(float(reward_terms.get("upright_score", 0.0)), 3),
                    "axes:",
                    (
                        round(float(reward_terms.get("trunk_x_up", 0.0)), 3),
                        round(float(reward_terms.get("trunk_y_up", 0.0)), 3),
                        round(float(reward_terms.get("trunk_z_up", 0.0)), 3),
                    ),
                )
            step += 1
            if terminated or truncated:
                obs, _ = env.reset()
                step = 0
    finally:
        env.close()


if __name__ == "__main__":
    main()
