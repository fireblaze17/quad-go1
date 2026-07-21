"""Train a standing policy for the Chrono Go1 environment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from go1_env import Go1Env, standing_env_metadata
from project_config import DEFAULT_CHECKPOINT_FREQ, DEFAULT_MAX_STEPS, SB3_DEVICE


RESET_NOISE_LEVELS = ("clean", "rn1", "rn2", "rn3")
RESET_NOISE_COMPONENTS = ("combined", "joint_pos", "joint_vel", "roll_pitch", "yaw", "base_height", "base_position", "base_velocity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.8)
    parser.add_argument("--friction-max", type=float, default=0.8)
    parser.add_argument("--ground-height-offset", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/stand_v4_implicit_limited_drive_reward_aligned_1m"))
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reset-noise-level", choices=RESET_NOISE_LEVELS, default="clean")
    parser.add_argument("--reset-noise-components", choices=RESET_NOISE_COMPONENTS, default="combined")
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
    parser.add_argument(
        "--eval-during-training",
        action="store_true",
        help="Periodically run deterministic eval, save best_model.zip, and write a learning curve.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Evaluate every N environment steps when --eval-during-training is enabled.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of eval episodes per training-time evaluation.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many eval checkpoints without improvement. Set 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum eval mean reward improvement required to reset patience.",
    )
    parser.add_argument(
        "--eval-friction-min",
        type=float,
        default=None,
        help="Evaluation friction min. Defaults to --friction-min.",
    )
    parser.add_argument(
        "--eval-friction-max",
        type=float,
        default=None,
        help="Evaluation friction max. Defaults to --friction-max.",
    )
    parser.add_argument(
        "--eval-reset-noise-level",
        choices=RESET_NOISE_LEVELS,
        default=None,
        help="Evaluation reset-noise level. Defaults to --reset-noise-level.",
    )
    parser.add_argument(
        "--eval-reset-noise-components",
        choices=RESET_NOISE_COMPONENTS,
        default=None,
        help="Evaluation reset-noise components. Defaults to --reset-noise-components.",
    )
    return parser.parse_args()


def make_env(args):
    env = Go1Env(
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(args.friction_min, args.friction_max),
        reset_noise_level=args.reset_noise_level,
        reset_noise_components=args.reset_noise_components,
        ground_height_offset=args.ground_height_offset,
    )
    return Monitor(env)


def make_eval_env(args):
    env = Go1Env(
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(
            args.friction_min if args.eval_friction_min is None else args.eval_friction_min,
            args.friction_max if args.eval_friction_max is None else args.eval_friction_max,
        ),
        reset_noise_level=args.reset_noise_level
        if args.eval_reset_noise_level is None
        else args.eval_reset_noise_level,
        reset_noise_components=args.reset_noise_components
        if args.eval_reset_noise_components is None
        else args.eval_reset_noise_components,
        ground_height_offset=args.ground_height_offset,
    )
    return Monitor(env)


class TrainingEvalEarlyStopCallback(BaseCallback):
    """Save eval learning curves and stop after eval reward patience expires."""

    def __init__(
        self,
        eval_env,
        save_dir: Path,
        eval_freq: int,
        eval_episodes: int,
        patience: int,
        min_delta: float,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.eval_freq = max(1, int(eval_freq))
        self.eval_episodes = max(1, int(eval_episodes))
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.best_eval_reward = -float("inf")
        self.best_step = 0
        self.no_improve_count = 0
        self.last_eval_step = 0
        self.last_train_mean_reward: float | None = None
        self.last_eval_mean_reward: float | None = None
        self.rows: list[dict] = []
        self.curve_csv = save_dir / "learning_curve.csv"
        self.curve_json = save_dir / "learning_curve.json"
        self.curve_png = save_dir / "learning_curve.png"
        self.best_model_path = save_dir / "best_model"
        (save_dir / "eval_checkpoints").mkdir(parents=True, exist_ok=True)

    def _train_mean_reward(self) -> float:
        if len(self.model.ep_info_buffer) == 0:
            return float("nan")
        return float(np.mean([episode["r"] for episode in self.model.ep_info_buffer]))

    def _run_eval(self) -> tuple[float, float, float]:
        rewards = []
        lengths = []
        for _ in range(self.eval_episodes):
            obs, _info = self.eval_env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = self.eval_env.step(action)
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            rewards.append(total_reward)
            lengths.append(steps)
        return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(lengths))

    def _write_curve(self) -> None:
        fieldnames = [
            "timesteps",
            "train_mean_reward",
            "eval_mean_reward",
            "eval_std_reward",
            "eval_mean_length",
            "best_eval_reward",
            "best_step",
            "no_improve_count",
            "overfit_warning",
        ]
        with self.curve_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        self.curve_json.write_text(json.dumps(self.rows, indent=2) + "\n", encoding="utf-8")
        self._write_curve_plot()

    def _write_curve_plot(self) -> None:
        if not self.rows:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            if len(self.rows) == 1:
                print(f"warning: could not import matplotlib for learning curve plot: {exc}")
            return

        steps = [row["timesteps"] for row in self.rows]
        train_rewards = [row["train_mean_reward"] for row in self.rows]
        eval_rewards = [row["eval_mean_reward"] for row in self.rows]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, train_rewards, marker="o", label="train mean reward")
        ax.plot(steps, eval_rewards, marker="o", label="eval mean reward")
        ax.axvline(self.best_step, color="tab:green", linestyle="--", linewidth=1, label="best eval")
        ax.set_xlabel("environment timesteps")
        ax.set_ylabel("reward")
        ax.set_title("Training/Evaluation Learning Curve")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.curve_png, dpi=150)
        plt.close(fig)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps

        train_mean_reward = self._train_mean_reward()
        eval_mean_reward, eval_std_reward, eval_mean_length = self._run_eval()
        improved = eval_mean_reward > self.best_eval_reward + self.min_delta
        if improved:
            self.best_eval_reward = eval_mean_reward
            self.best_step = self.num_timesteps
            self.no_improve_count = 0
            self.model.save(self.best_model_path)
        else:
            self.no_improve_count += 1

        self.model.save(
            self.save_dir / "eval_checkpoints" / f"stand_policy_{self.num_timesteps}_steps"
        )

        overfit_warning = False
        if self.last_train_mean_reward is not None and self.last_eval_mean_reward is not None:
            train_up = train_mean_reward > self.last_train_mean_reward + self.min_delta
            eval_down = eval_mean_reward < self.last_eval_mean_reward - self.min_delta
            overfit_warning = bool(train_up and eval_down)

        row = {
            "timesteps": self.num_timesteps,
            "train_mean_reward": train_mean_reward,
            "eval_mean_reward": eval_mean_reward,
            "eval_std_reward": eval_std_reward,
            "eval_mean_length": eval_mean_length,
            "best_eval_reward": self.best_eval_reward,
            "best_step": self.best_step,
            "no_improve_count": self.no_improve_count,
            "overfit_warning": overfit_warning,
        }
        self.rows.append(row)
        self._write_curve()

        self.last_train_mean_reward = train_mean_reward
        self.last_eval_mean_reward = eval_mean_reward

        print(
            "eval "
            f"step={self.num_timesteps} "
            f"train_mean_reward={train_mean_reward:.3f} "
            f"eval_mean_reward={eval_mean_reward:.3f} "
            f"best={self.best_eval_reward:.3f}@{self.best_step} "
            f"patience={self.no_improve_count}/{self.patience}"
        )

        if self.patience > 0 and self.no_improve_count >= self.patience:
            print(
                "early stopping: eval reward did not improve for "
                f"{self.no_improve_count} eval checkpoints"
            )
            return False
        return True

    def _on_training_end(self) -> None:
        self._write_curve()
        self.eval_env.close()


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


def save_eval_metadata(args) -> None:
    if not args.eval_during_training:
        return
    metadata = {
        "enabled": True,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "eval_friction_min": args.friction_min
        if args.eval_friction_min is None
        else args.eval_friction_min,
        "eval_friction_max": args.friction_max
        if args.eval_friction_max is None
        else args.eval_friction_max,
        "eval_reset_noise_level": args.reset_noise_level
        if args.eval_reset_noise_level is None
        else args.eval_reset_noise_level,
        "eval_reset_noise_components": args.reset_noise_components
        if args.eval_reset_noise_components is None
        else args.eval_reset_noise_components,
        "selection_rule": "best eval_mean_reward with patience on no improvement",
        "notes": (
            "Use best_model.zip as the reward-selected checkpoint, then run "
            "diagnose_policy.py gate checks before promotion."
        ),
    }
    (args.save_dir / "training_eval.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
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
    save_eval_metadata(args)

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

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(args.save_dir / "checkpoints"),
                name_prefix="stand_policy",
            )
        )
    if args.eval_during_training:
        if args.eval_freq <= 0:
            raise ValueError("--eval-freq must be positive when --eval-during-training is enabled")
        callbacks.append(
            TrainingEvalEarlyStopCallback(
                eval_env=make_eval_env(args),
                save_dir=args.save_dir,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
        )
    callback = None if not callbacks else CallbackList(callbacks)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save_dir / "final_model")
    env.close()


if __name__ == "__main__":
    main()
