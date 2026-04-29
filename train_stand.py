"""Train a standing policy for the Chrono Go1 environment."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from go1_env import Go1Env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.8)
    parser.add_argument("--friction-max", type=float, default=0.8)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/stand"))
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def make_env(args):
    env = Go1Env(
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(args.friction_min, args.friction_max),
    )
    return Monitor(env)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    if args.load is not None:
        model = PPO.load(args.load, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
        )

    checkpoint = CheckpointCallback(
        save_freq=25_000,
        save_path=str(args.save_dir / "checkpoints"),
        name_prefix="stand_policy",
    )
    model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    model.save(args.save_dir / "final_model")
    env.close()


if __name__ == "__main__":
    main()
