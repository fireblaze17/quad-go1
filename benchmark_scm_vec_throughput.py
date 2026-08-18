"""Benchmark flat vs SCM vectorized environment stepping throughput."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from project_config import CURRENT_BASELINE_MODEL, SB3_DEVICE


ACTUATOR_MODELS = ("actuator_net", "torque_limited_pd")
ACTION_SOURCES = ("policy", "zero", "random")
ENV_BACKENDS = ("flat", "scm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-backends", choices=ENV_BACKENDS, nargs="+", default=["flat", "scm"])
    parser.add_argument("--env-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24])
    parser.add_argument("--out", type=Path, default=Path("outputs/scm_benchmarks/scm_vec_throughput.csv"))
    parser.add_argument("--policy", type=Path, default=CURRENT_BASELINE_MODEL)
    parser.add_argument("--action-source", choices=ACTION_SOURCES, default="policy")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--actuator-model", choices=ACTUATOR_MODELS, default="actuator_net")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--fixed-command-vx", type=float, default=-0.5)
    parser.add_argument("--fixed-command-vz", type=float, default=0.0)
    parser.add_argument("--fixed-command-yaw-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vec-start-method", choices=("fork", "forkserver", "spawn"), default="fork")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=SB3_DEVICE)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--random-action-abs", type=float, default=1.0)
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument("--no-observation-noise", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def _set_thread_defaults(torch_threads: int) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    try:
        import torch

        torch.set_num_threads(max(1, int(torch_threads)))
    except Exception:
        pass


def _env_class(env_backend: str):
    if env_backend == "scm":
        from go1_scm_env import Go1SCMEnv

        return Go1SCMEnv
    if env_backend == "flat":
        from go1_env import Go1Env

        return Go1Env
    raise ValueError(f"unsupported env backend: {env_backend}")


def _make_env(
    *,
    env_backend: str,
    max_steps: int,
    fixed_command: tuple[float, float, float],
    actuator_model: str,
    seed: int,
    rank: int,
    default_randomization: bool,
    observation_noise: bool,
):
    env_cls = _env_class(env_backend)
    env = env_cls(
        max_steps=max_steps,
        enable_motors=True,
        fixed_command=fixed_command,
        command_seed=seed + 1000 * rank,
        env_rank=rank,
        actuator_model=actuator_model,
        default_randomization=default_randomization,
        observation_noise=observation_noise,
        visual_mesh_format="none",
    )
    env.reset(seed=seed + rank)
    return env


def _make_vec_env(args: argparse.Namespace, env_backend: str, num_envs: int):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    fixed_command = (
        float(args.fixed_command_vx),
        float(args.fixed_command_vz),
        float(args.fixed_command_yaw_rate),
    )
    env_fns = [
        partial(
            _make_env,
            env_backend=env_backend,
            max_steps=args.max_steps,
            fixed_command=fixed_command,
            actuator_model=args.actuator_model,
            seed=args.seed,
            rank=rank,
            default_randomization=not args.no_randomization,
            observation_noise=not args.no_observation_noise,
        )
        for rank in range(num_envs)
    ]

    if num_envs == 1:
        vec_env = DummyVecEnv(env_fns)
        vec_mode = "dummy_single_env"
    else:
        vec_env = SubprocVecEnv(env_fns, start_method=args.vec_start_method)
        vec_mode = f"subproc_{args.vec_start_method}"
    return VecMonitor(vec_env), vec_mode


def _actions(args: argparse.Namespace, model, obs: np.ndarray, vec_env, rng: np.random.Generator) -> np.ndarray:
    if args.action_source == "policy":
        actions, _ = model.predict(obs, deterministic=not args.stochastic)
        return np.asarray(actions, dtype=np.float32)
    shape = (vec_env.num_envs, *vec_env.action_space.shape)
    if args.action_source == "zero":
        return np.zeros(shape, dtype=np.float32)
    return rng.uniform(
        -float(args.random_action_abs),
        float(args.random_action_abs),
        size=shape,
    ).astype(np.float32)


def run_one(args: argparse.Namespace, env_backend: str, num_envs: int) -> dict[str, Any]:
    _set_thread_defaults(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    vec_env = None
    started = time.perf_counter()
    try:
        vec_env, vec_mode = _make_vec_env(args, env_backend, num_envs)
        obs = vec_env.reset()
        model = None
        if args.action_source == "policy":
            from ppo_compat import EnvClippedActionPPO, load_ppo_same_shape_action_space

            model = load_ppo_same_shape_action_space(
                EnvClippedActionPPO,
                args.policy,
                env=vec_env,
                device=args.device,
            )

        for _ in range(max(0, int(args.warmup_steps))):
            action = _actions(args, model, obs, vec_env, rng)
            obs, _rewards, _dones, _infos = vec_env.step(action)

        measured_start = time.perf_counter()
        total_reward = 0.0
        done_count = 0
        for _ in range(max(1, int(args.steps))):
            action = _actions(args, model, obs, vec_env, rng)
            obs, rewards, dones, _infos = vec_env.step(action)
            total_reward += float(np.sum(rewards))
            done_count += int(np.sum(dones))
        measured_wall = time.perf_counter() - measured_start

        env_steps = int(args.steps) * int(num_envs)
        simulated_seconds_per_env = float(args.steps) * 0.02
        aggregate_simulated_seconds = float(env_steps) * 0.02
        return {
            "status": "ok",
            "env_backend": env_backend,
            "num_envs": int(num_envs),
            "vec_mode": vec_mode,
            "action_source": args.action_source,
            "actuator_model": args.actuator_model,
            "steps_per_env": int(args.steps),
            "warmup_steps_per_env": int(args.warmup_steps),
            "wall_seconds": measured_wall,
            "startup_plus_warmup_seconds": measured_start - started,
            "env_steps": env_steps,
            "policy_batches_per_second": float(args.steps) / measured_wall,
            "env_steps_per_second": float(env_steps) / measured_wall,
            "per_env_simulated_seconds": simulated_seconds_per_env,
            "aggregate_simulated_seconds": aggregate_simulated_seconds,
            "per_env_realtime_factor": simulated_seconds_per_env / measured_wall,
            "aggregate_realtime_factor": aggregate_simulated_seconds / measured_wall,
            "mean_reward_per_env_step": total_reward / max(env_steps, 1),
            "done_count": done_count,
            "device": args.device,
            "torch_threads": int(args.torch_threads),
            "vec_start_method": args.vec_start_method,
            "default_randomization": not args.no_randomization,
            "observation_noise": not args.no_observation_noise,
        }
    finally:
        if vec_env is not None:
            vec_env.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(args.repeats))):
        for env_backend in args.env_backends:
            for num_envs in args.env_counts:
                print(
                    f"\n=== Benchmark repeat={repeat} backend={env_backend} num_envs={num_envs} ===",
                    flush=True,
                )
                row = run_one(args, env_backend, int(num_envs))
                row["repeat"] = repeat
                rows.append(row)
                print(json.dumps(row, indent=2), flush=True)
                _write_csv(args.out, rows)
    print(f"\nWrote benchmark CSV: {args.out}", flush=True)


if __name__ == "__main__":
    main()
