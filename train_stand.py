"""Train the default Chrono Go1 locomotion policy."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from go1_env import Go1Env, _TIME_STEP, standing_env_metadata
from ppo_compat import EnvClippedActionPPO, load_ppo_same_shape_action_space
from project_config import DEFAULT_CHECKPOINT_FREQ, DEFAULT_MAX_STEPS, SB3_DEVICE


ACTUATOR_MODELS = ("actuator_net", "torque_limited_pd")
ENV_BACKENDS = ("flat", "scm")


def env_class_for_backend(env_backend: str):
    if env_backend == "scm":
        from go1_scm_env import Go1SCMEnv

        return Go1SCMEnv
    if env_backend == "flat":
        return Go1Env
    raise ValueError(f"unsupported env backend: {env_backend}")


def metadata_for_backend(env_backend: str, actuator_model: str) -> dict:
    metadata = standing_env_metadata(actuator_model)
    metadata["env_backend"] = env_backend
    if env_backend == "scm":
        from go1_scm_env import scm_env_metadata

        metadata["terrain"] = "scm"
        metadata["scm"] = scm_env_metadata()
        metadata["physics_time_step"] = metadata["scm"]["scm_physics_dt"]
        metadata["physics_frequency"] = 1.0 / metadata["scm"]["scm_physics_dt"]
        metadata["physics_substeps"] = metadata["scm"]["scm_substeps"]
        metadata["solver"] = {
            "type": metadata["scm"]["scm_solver_type"],
            "max_iterations": metadata["scm"]["scm_solver_iterations"],
        }
    return metadata


class RolloutDiagnosticsTensorboardCallback(BaseCallback):
    """Log default rollout diagnostics to TensorBoard."""

    foot_order = ("FR", "FL", "RR", "RL")

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.samples: list[dict[str, float]] = []
        self.sampled_action_rms_values: list[float] = []
        self.policy_mean_rms_values: list[float] = []

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _rms(array: np.ndarray) -> float:
        if array.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(array.astype(np.float64)))))

    def _record_action_distribution_stats(self) -> None:
        actions = self.locals.get("actions", None)
        if actions is not None:
            action_array = np.asarray(actions, dtype=np.float32)
            if action_array.size > 0:
                self.sampled_action_rms_values.append(self._rms(action_array))

        obs_tensor = self.locals.get("obs_tensor", None)
        if obs_tensor is None:
            return
        try:
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(obs_tensor)
                torch_dist = getattr(distribution, "distribution", None)
                mean_actions = getattr(torch_dist, "mean", None)
                if mean_actions is not None:
                    self.policy_mean_rms_values.append(
                        float(torch.sqrt(torch.mean(mean_actions.float() ** 2)).detach().cpu().item())
                    )
        except Exception:
            return

    def _collect_samples(self) -> None:
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            terms = info.get("reward_terms", {})
            if not isinstance(terms, dict):
                terms = {}
            lin_err_x = self._safe_float(terms.get("lin_vel_error_x", 0.0))
            lin_err_z = self._safe_float(terms.get("lin_vel_error_z", 0.0))
            sample = {
                "env_idx": float(env_idx),
                "body_vx": self._safe_float(terms.get("body_lin_vel_x", 0.0)),
                "body_vz": self._safe_float(terms.get("body_lin_vel_z", 0.0)),
                "body_yaw_rate": self._safe_float(terms.get("body_yaw_rate", 0.0)),
                "trunk_yaw": self._safe_float(terms.get("trunk_yaw", 0.0)),
                "lin_abs_tracking_error": float(math.sqrt(lin_err_x ** 2 + lin_err_z ** 2)),
                "vx_abs_tracking_error": abs(lin_err_x),
                "base_world_x": self._safe_float(terms.get("base_world_x", 0.0)),
                "base_world_z": self._safe_float(terms.get("base_world_z", 0.0)),
                "reward_raw_sum": self._safe_float(terms.get("reward_raw_sum", 0.0)),
                "tracking_lin_vel_reward": self._safe_float(terms.get("tracking_lin_vel_reward", 0.0)),
                "tracking_ang_vel_reward": self._safe_float(terms.get("tracking_ang_vel_reward", 0.0)),
                "lin_vel_z_reward": self._safe_float(terms.get("lin_vel_z_reward", 0.0)),
                "ang_vel_xy_reward": self._safe_float(terms.get("ang_vel_xy_reward", 0.0)),
                "flat_orientation_l2_reward": self._safe_float(terms.get("flat_orientation_l2_reward", 0.0)),
                "dof_acc_reward": self._safe_float(terms.get("dof_acc_reward", 0.0)),
                "action_rate_reward": self._safe_float(terms.get("action_rate_reward", 0.0)),
                "torques_reward": self._safe_float(terms.get("torques_reward", 0.0)),
                "feet_air_time_reward": self._safe_float(terms.get("feet_air_time_reward", 0.0)),
                "preclip_policy_action_abs_mean": self._safe_float(terms.get("preclip_policy_action_abs_mean", 0.0)),
                "preclip_policy_action_abs_max": self._safe_float(terms.get("preclip_policy_action_abs_max", 0.0)),
            }
            for foot in self.foot_order:
                sample[f"foot_contact_{foot}"] = self._safe_float(terms.get(f"foot_contact_{foot}", 0.0))
            self.samples.append(sample)

    @staticmethod
    def _run_durations(states: list[bool], desired_state: bool) -> list[float]:
        durations: list[float] = []
        current = 0
        for state in states:
            if state == desired_state:
                current += 1
            elif current:
                durations.append(current * _TIME_STEP)
                current = 0
        if current:
            durations.append(current * _TIME_STEP)
        return durations

    def _record_contact_metrics(self) -> None:
        if not self.samples:
            return
        env_ids = sorted({int(sample["env_idx"]) for sample in self.samples})
        total_switches_by_foot: dict[str, float] = {}
        for foot in self.foot_order:
            switch_counts = []
            stance_durations = []
            swing_durations = []
            contact_fractions = []
            for env_idx in env_ids:
                states = [
                    sample[f"foot_contact_{foot}"] > 0.5
                    for sample in self.samples
                    if int(sample["env_idx"]) == env_idx
                ]
                if not states:
                    continue
                switch_counts.append(float(np.sum(np.asarray(states[1:]) != np.asarray(states[:-1]))))
                contact_fractions.append(float(np.mean(states)))
                stance_durations.extend(self._run_durations(states, True))
                swing_durations.extend(self._run_durations(states, False))
            if switch_counts:
                total_switches_by_foot[foot] = float(np.sum(switch_counts))
                self.logger.record(f"feet/contact_switches_{foot}_mean_per_env", float(np.mean(switch_counts)))
                self.logger.record(f"feet/contact_fraction_{foot}", float(np.mean(contact_fractions)))
            if stance_durations:
                self.logger.record(f"feet/stance_duration_{foot}_mean_s", float(np.mean(stance_durations)))
            if swing_durations:
                self.logger.record(f"feet/swing_duration_{foot}_mean_s", float(np.mean(swing_durations)))
        total_switches = float(sum(total_switches_by_foot.values()))
        if total_switches > 0.0:
            for foot, count in total_switches_by_foot.items():
                self.logger.record(f"feet/contact_switch_share_{foot}", float(count / total_switches))
            self.logger.record("feet/max_single_foot_switch_share", float(max(total_switches_by_foot.values()) / total_switches))

    def _record_rollout_logs(self) -> None:
        if not self.samples:
            return

        def values(key: str) -> np.ndarray:
            return np.asarray([sample[key] for sample in self.samples], dtype=np.float64)

        body_vx = values("body_vx")
        self.logger.record("kinematics/body_vx_mean", float(np.mean(body_vx)))
        self.logger.record("kinematics/body_vx_median", float(np.median(body_vx)))
        self.logger.record("tracking/lin_abs_error_mean", float(np.mean(values("lin_abs_tracking_error"))))
        self.logger.record("tracking/vx_abs_error_mean", float(np.mean(values("vx_abs_tracking_error"))))
        for key, tag in (
            ("tracking_lin_vel_reward", "reward_terms/tracking_lin_vel_reward_mean"),
            ("tracking_ang_vel_reward", "reward_terms/tracking_ang_vel_reward_mean"),
            ("lin_vel_z_reward", "reward_terms/lin_vel_z_reward_mean"),
            ("ang_vel_xy_reward", "reward_terms/ang_vel_xy_reward_mean"),
            ("flat_orientation_l2_reward", "reward_terms/flat_orientation_l2_reward_mean"),
            ("dof_acc_reward", "reward_terms/dof_acc_reward_mean"),
            ("action_rate_reward", "reward_terms/action_rate_reward_mean"),
            ("torques_reward", "reward_terms/torques_reward_mean"),
            ("feet_air_time_reward", "reward_terms/feet_air_time_reward_mean"),
        ):
            self.logger.record(tag, float(np.mean(values(key))))
        self.logger.record("policy/preclip_action_abs_mean", float(np.mean(values("preclip_policy_action_abs_mean"))))
        self.logger.record("policy/preclip_action_abs_max", float(np.max(values("preclip_policy_action_abs_max"))))
        if self.policy_mean_rms_values:
            self.logger.record("policy/mean_action_rms", float(np.mean(self.policy_mean_rms_values)))
        if self.sampled_action_rms_values:
            self.logger.record("policy/sampled_action_rms", float(np.mean(self.sampled_action_rms_values)))
        policy = getattr(self.model, "policy", None)
        if policy is not None and hasattr(policy, "log_std"):
            std = torch.exp(policy.log_std).detach().cpu().float().numpy()
            self.logger.record("policy/std_mean", float(np.mean(std)))
            self.logger.record("policy/std_min", float(np.min(std)))
            self.logger.record("policy/std_max", float(np.max(std)))
        self._record_contact_metrics()
        self.samples = []
        self.sampled_action_rms_values = []
        self.policy_mean_rms_values = []

    def _on_step(self) -> bool:
        self._record_action_distribution_stats()
        self._collect_samples()
        return True

    def _on_rollout_end(self) -> None:
        self._record_rollout_logs()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--target-total-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--save-dir", "--run-dir", dest="save_dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-state", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fixed-command-vx", type=float, default=0.0)
    parser.add_argument("--fixed-command-vz", type=float, default=0.0)
    parser.add_argument("--fixed-command-yaw-rate", type=float, default=0.0)
    parser.add_argument("--env-backend", choices=ENV_BACKENDS, default="flat")
    parser.add_argument("--actuator-model", choices=ACTUATOR_MODELS, default="actuator_net")
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--vec-start-method", choices=["fork", "forkserver", "spawn"], default="forkserver")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=SB3_DEVICE)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1536)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-final", type=float, default=1e-4)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--eval-during-training", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    return parser.parse_args()


def _set_thread_defaults(torch_threads: int) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    torch.set_num_threads(max(1, int(torch_threads)))


def make_env(args, rank: int = 0, monitor: bool = True):
    env_cls = env_class_for_backend(args.env_backend)
    env = env_cls(
        max_steps=args.max_steps,
        enable_motors=True,
        fixed_command=(args.fixed_command_vx, args.fixed_command_vz, args.fixed_command_yaw_rate),
        command_seed=args.seed + 1000 * rank,
        env_rank=rank,
        actuator_model=args.actuator_model,
        visual_mesh_format="none",
    )
    env.reset(seed=args.seed + rank)
    return Monitor(env) if monitor else env


def make_train_env(args):
    env_fns = [lambda rank=rank: make_env(args, rank, monitor=False) for rank in range(max(1, int(args.num_envs)))]
    use_dummy = len(env_fns) == 1
    if use_dummy:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method=args.vec_start_method)
    return VecMonitor(env)


def make_eval_env(args):
    return Monitor(make_env(args, rank=999, monitor=False))


def tensorboard_log_dir(args) -> Path:
    return args.save_dir / "tensorboard"


def make_learning_rate_schedule(args):
    if float(args.learning_rate_final) == float(args.learning_rate):
        return float(args.learning_rate)

    initial = float(args.learning_rate)
    final = float(args.learning_rate_final)

    def _schedule(progress_remaining: float) -> float:
        return final + float(progress_remaining) * (initial - final)

    return _schedule


def default_policy_kwargs() -> dict[str, Any]:
    return {
        "net_arch": {"pi": [512, 256, 128], "vf": [512, 256, 128]},
        "activation_fn": torch.nn.ELU,
        "log_std_init": math.log(1.0),
    }


def initialize_action_head(model: EnvClippedActionPPO, loaded: bool) -> None:
    if loaded or not hasattr(model.policy, "action_net"):
        return
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()


def apply_ppo_overrides(model: EnvClippedActionPPO, args) -> None:
    model.learning_rate = make_learning_rate_schedule(args)
    model.lr_schedule = get_schedule_fn(model.learning_rate)
    model.clip_range = lambda _: float(args.clip_range)
    model.target_kl = args.target_kl
    model.n_steps = int(args.n_steps)
    model.batch_size = int(args.batch_size)
    model.n_epochs = int(args.n_epochs)
    model.ent_coef = float(args.ent_coef)
    model.vf_coef = float(args.vf_coef)
    model.gae_lambda = float(args.gae_lambda)
    model.max_grad_norm = float(args.max_grad_norm)
    model.tensorboard_log = str(tensorboard_log_dir(args))
    model.set_random_seed(args.seed)
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )


def load_resume_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_run_metadata(args) -> None:
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args_json = vars(args).copy()
    for key in ("save_dir", "load", "resume_model", "resume_state"):
        args_json[key] = None if args_json[key] is None else str(args_json[key])
    env_metadata = metadata_for_backend(args.env_backend, args.actuator_model)
    (args.save_dir / "args.json").write_text(json.dumps(args_json, indent=2) + "\n", encoding="utf-8")
    (args.save_dir / "env_constants.json").write_text(json.dumps(env_metadata, indent=2) + "\n", encoding="utf-8")
    (args.save_dir / "config.json").write_text(
        json.dumps({"args": args_json, "env": env_metadata}, indent=2) + "\n",
        encoding="utf-8",
    )


class TrainingEvalEarlyStopCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, eval_episodes: int, patience: int, min_delta: float) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.eval_episodes = max(1, int(eval_episodes))
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.best_eval_reward = -float("inf")
        self.no_improve_count = 0
        self.last_eval_step = 0

    def _run_eval(self) -> float:
        rewards = []
        for _ in range(self.eval_episodes):
            obs, _info = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = self.eval_env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
            rewards.append(total_reward)
        return float(np.mean(rewards))

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps
        eval_reward = self._run_eval()
        if eval_reward > self.best_eval_reward + self.min_delta:
            self.best_eval_reward = eval_reward
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        print(
            f"eval step={self.num_timesteps} eval_mean_reward={eval_reward:.3f} "
            f"best={self.best_eval_reward:.3f} patience={self.no_improve_count}/{self.patience}",
            flush=True,
        )
        return not (self.patience > 0 and self.no_improve_count >= self.patience)

    def _on_training_end(self) -> None:
        self.eval_env.close()


def main() -> None:
    args = parse_args()
    args.num_envs = max(1, int(args.num_envs))
    if args.resume_model is not None and args.load is not None:
        raise ValueError("Use either --load or --resume-model, not both.")
    if args.resume_state is not None and args.resume_model is None:
        raise ValueError("--resume-state requires --resume-model.")

    resume_state = load_resume_state(args.resume_state)
    start_steps = int(resume_state.get("num_timesteps", 0)) if resume_state else 0
    target_total_specified = args.target_total_steps is not None
    if args.target_total_steps is None:
        args.target_total_steps = args.timesteps

    _set_thread_defaults(args.torch_threads)
    save_run_metadata(args)
    (args.save_dir / "startup_status.txt").write_text("metadata_saved\n", encoding="utf-8")

    print(
        f"[startup] building train envs num_envs={args.num_envs} "
        f"vec_start_method={args.vec_start_method} backend={args.env_backend} actuator={args.actuator_model}",
        flush=True,
    )
    env = make_train_env(args)
    print("[startup] train envs ready; building/loading PPO model", flush=True)

    load_path = args.resume_model if args.resume_model is not None else args.load
    loaded = load_path is not None
    if loaded:
        model = load_ppo_same_shape_action_space(EnvClippedActionPPO, load_path, env=env, device=args.device)
        apply_ppo_overrides(model, args)
        if resume_state is not None:
            model.num_timesteps = start_steps
        elif args.resume_model is not None:
            start_steps = int(getattr(model, "num_timesteps", 0))
    else:
        model = EnvClippedActionPPO(
            "MlpPolicy",
            env,
            device=args.device,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(tensorboard_log_dir(args)),
            policy_kwargs=default_policy_kwargs(),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=make_learning_rate_schedule(args),
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )
    initialize_action_head(model, loaded)

    additional_steps = (
        int(args.target_total_steps) - start_steps
        if args.resume_model is not None or target_total_specified
        else int(args.timesteps)
    )
    if additional_steps <= 0:
        raise ValueError(f"No timesteps left to train: target={args.target_total_steps}, resume={start_steps}")

    callbacks: list[BaseCallback] = [RolloutDiagnosticsTensorboardCallback()]
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, int(args.checkpoint_freq // args.num_envs)),
                save_path=str(args.save_dir / "checkpoints"),
                name_prefix="stand_policy",
            )
        )
    if args.eval_during_training:
        callbacks.append(
            TrainingEvalEarlyStopCallback(
                eval_env=make_eval_env(args),
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
        )

    model.learn(
        total_timesteps=additional_steps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=args.resume_model is None,
    )
    model.save(args.save_dir / "final_model")
    final_state = {
        "stack": "default",
        "num_timesteps": int(model.num_timesteps),
        "actuator_model": args.actuator_model,
        "friction": 0.8,
        "command_sampler": "default",
        "randomization": "default",
    }
    (args.save_dir / "final_model.state.json").write_text(json.dumps(final_state, indent=2) + "\n", encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
