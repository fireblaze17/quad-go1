"""Train a standing policy for the Chrono Go1 environment."""

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from go1_env import Go1Env, standing_env_metadata
from project_config import DEFAULT_CHECKPOINT_FREQ, SB3_DEVICE


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
    parser.add_argument(
        "--action-filter-tau",
        type=float,
        default=None,
        help="Optional environment action low-pass filter time constant in seconds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate. Also overrides loaded models for fine-tuning.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range. Also overrides loaded models for fine-tuning.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="PPO target KL early-stopping threshold.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=DEFAULT_CHECKPOINT_FREQ,
        help="Save PPO checkpoints every N environment steps. Set <=0 to disable.",
    )
    return parser.parse_args()


def make_env(args):
    env = Go1Env(
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(args.friction_min, args.friction_max),
        action_filter_tau=args.action_filter_tau,
    )
    return Monitor(env)


def _json_ready_args(args) -> dict:
    data = vars(args).copy()
    data["save_dir"] = str(data["save_dir"])
    data["load"] = None if data["load"] is None else str(data["load"])
    return data


def save_run_metadata(args) -> None:
    (args.save_dir / "args.json").write_text(
        json.dumps(_json_ready_args(args), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.save_dir / "env_constants.json").write_text(
        json.dumps(standing_env_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )


def apply_ppo_overrides(model: PPO, args) -> None:
    model.learning_rate = args.learning_rate
    model.lr_schedule = get_schedule_fn(args.learning_rate)
    model.clip_range = get_schedule_fn(args.clip_range)
    model.target_kl = args.target_kl
    model.set_random_seed(args.seed)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(args)

    env = make_env(args)
    if args.load is not None:
        model = PPO.load(args.load, env=env, device=SB3_DEVICE)
        apply_ppo_overrides(model, args)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device=SB3_DEVICE,
            verbose=1,
            seed=args.seed,
            n_steps=1024,
            batch_size=256,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            gamma=0.99,
        )

    callback = None
    if args.checkpoint_freq > 0:
        callback = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(args.save_dir / "checkpoints"),
            name_prefix="stand_policy",
        )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save_dir / "final_model")
    env.close()


if __name__ == "__main__":
    main()
