"""Headless diagnostics for the default Chrono Go1 policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from diagnostics import LEG_PREFIXES, contact_body_groups, contact_debug_stats, foot_bodies, foot_debug_stats, foot_xz_positions
from go1_env import Go1Env, _TIME_STEP
from ppo_compat import EnvClippedActionPPO, load_ppo_same_shape_action_space
from project_config import CURRENT_BASELINE_MODEL, SB3_DEVICE


ACTUATOR_MODELS = ("actuator_net", "torque_limited_pd")
ENV_BACKENDS = ("flat", "scm")
ACTIVE_REWARD_TERMS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "ang_vel_xy",
    "torques",
    "dof_acc",
    "flat_orientation_l2",
    "feet_air_time",
    "action_rate",
)


def env_class_for_backend(env_backend: str):
    if env_backend == "scm":
        from go1_scm_env import Go1SCMEnv

        return Go1SCMEnv
    if env_backend == "flat":
        return Go1Env
    raise ValueError(f"unsupported env backend: {env_backend}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, nargs="?", default=CURRENT_BASELINE_MODEL)
    parser.add_argument("--fixed-command-vx", type=float, default=0.0)
    parser.add_argument("--fixed-command-vz", type=float, default=0.0)
    parser.add_argument("--fixed-command-yaw-rate", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-backend", choices=ENV_BACKENDS, default="flat")
    parser.add_argument("--actuator-model", choices=ACTUATOR_MODELS, default="actuator_net")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every-step", "--timeline", dest="log_every_step", action="store_true")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _episode_summary(
    episode_index: int,
    records: list[dict[str, Any]],
    total_reward: float,
    terminated: bool,
    truncated: bool,
    termination_reason: str,
) -> dict[str, Any]:
    if not records:
        return {
            "episode": episode_index,
            "length": 0,
            "reward": 0.0,
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": termination_reason,
        }

    def values(key: str) -> list[float]:
        return [_float(item.get(key, 0.0)) for item in records]

    contact_switches = {
        leg: int(np.sum(np.asarray(values(f"foot_contact_{leg}")[1:]) != np.asarray(values(f"foot_contact_{leg}")[:-1])))
        for leg in LEG_PREFIXES
    }
    reward_means = {
        f"{term}_reward_mean": _mean(values(f"{term}_reward"))
        for term in ACTIVE_REWARD_TERMS
    }
    weighted_means = {
        f"{term}_weighted_mean": _mean(values(f"{term}_weighted"))
        for term in ACTIVE_REWARD_TERMS
    }
    return {
        "episode": episode_index,
        "length": len(records),
        "reward": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": termination_reason,
        "mean_body_vx": _mean(values("body_lin_vel_x")),
        "mean_body_vz": _mean(values("body_lin_vel_z")),
        "mean_body_yaw_rate": _mean(values("body_yaw_rate")),
        "mean_abs_vx_error": _mean([abs(v) for v in values("lin_vel_error_x")]),
        "mean_abs_vz_error": _mean([abs(v) for v in values("lin_vel_error_z")]),
        "mean_abs_yaw_error": _mean([abs(v) for v in values("yaw_rate_error")]),
        "mean_reward_raw_sum": _mean(values("reward_raw_sum")),
        "mean_reward_dt_scaled": _mean(values("reward_dt_scaled")),
        "mean_abs_action": _mean(values("mean_abs_action")),
        "max_abs_action": float(max(values("max_abs_action"), default=0.0)),
        "mean_abs_motor_torque": _mean(values("mean_abs_motor_torque")),
        "max_abs_motor_torque": float(max(values("max_abs_motor_torque"), default=0.0)),
        "mean_torque_limit_fraction": _mean(values("mean_torque_limit_fraction")),
        "max_torque_limit_fraction": float(max(values("max_torque_limit_fraction"), default=0.0)),
        "fraction_torque_saturated": _mean(values("fraction_torque_saturated")),
        "min_foot_load": float(min(values("min_foot_load"), default=0.0)),
        "max_load_imbalance": float(max(values("foot_load_imbalance"), default=0.0)),
        "contact_switches": contact_switches,
        "total_contact_switches": int(sum(contact_switches.values())),
        **reward_means,
        **weighted_means,
    }


def _record_step(
    episode: int,
    step: int,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    foot_stats: dict[str, Any],
    contact_stats: dict[str, Any],
) -> dict[str, Any]:
    terms = info.get("reward_terms", {})
    record: dict[str, Any] = {
        "episode": episode,
        "step": step,
        "time": step * _TIME_STEP,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": info.get("termination_reason") or "",
        "action_mean_abs": float(np.mean(np.abs(action))),
    }
    for key, value in terms.items():
        if isinstance(value, (int, float, np.integer, np.floating, str, bool)):
            record[key] = _json_ready(value)
    for leg, value in zip(LEG_PREFIXES, foot_stats.get("foot_load_shares", [])):
        record[f"foot_load_share_{leg}"] = float(value)
    for leg, value in zip(LEG_PREFIXES, foot_stats.get("foot_friction_usage", [])):
        record[f"foot_friction_usage_{leg}"] = float(value)
    for leg, value in zip(LEG_PREFIXES, contact_stats.get("group_loads", {}).get("foot", [])):
        record[f"foot_load_{leg}"] = float(value)
    for leg, value in zip(LEG_PREFIXES, contact_stats.get("group_loads", {}).get("calf", [])):
        record[f"calf_load_{leg}"] = float(value)
    for leg, value in zip(LEG_PREFIXES, contact_stats.get("group_loads", {}).get("thigh", [])):
        record[f"thigh_load_{leg}"] = float(value)
    return record


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    if not args.policy.exists():
        raise FileNotFoundError(f"Policy not found: {args.policy}")
    args.out.mkdir(parents=True, exist_ok=True)

    env_cls = env_class_for_backend(args.env_backend)
    env = env_cls(
        max_steps=args.max_steps,
        enable_motors=True,
        fixed_command=(args.fixed_command_vx, args.fixed_command_vz, args.fixed_command_yaw_rate),
        command_seed=args.seed,
        actuator_model=args.actuator_model,
    )
    model = load_ppo_same_shape_action_space(EnvClippedActionPPO, args.policy, env=env, device=SB3_DEVICE)

    episodes: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    reset_infos: list[dict[str, Any]] = []
    for episode in range(max(1, int(args.episodes))):
        obs, reset_info = env.reset(seed=args.seed + episode)
        reset_infos.append(_json_ready(reset_info))
        tracked_feet = foot_bodies(env)
        tracked_contacts = contact_body_groups(env)
        reset_foot_xz = foot_xz_positions(tracked_feet)
        records: list[dict[str, Any]] = []
        total_reward = 0.0
        terminated = False
        truncated = False
        termination_reason = ""
        for step in range(max(1, int(args.max_steps))):
            action, _state = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            foot_stats = foot_debug_stats(tracked_feet, reset_foot_xz)
            contact_stats = contact_debug_stats(tracked_contacts)
            row = _record_step(episode, step, np.asarray(action), float(reward), terminated, truncated, info, foot_stats, contact_stats)
            records.append(row)
            if args.log_every_step:
                timeline.append(row)
            total_reward += float(reward)
            termination_reason = info.get("termination_reason") or ""
            if terminated or truncated:
                break
        episodes.append(_episode_summary(episode, records, total_reward, terminated, truncated, termination_reason))

    summary = {
        "policy": str(args.policy),
        "env_backend": args.env_backend,
        "actuator_model": args.actuator_model,
        "fixed_command": {
            "vx": float(args.fixed_command_vx),
            "vz": float(args.fixed_command_vz),
            "yaw_rate": float(args.fixed_command_yaw_rate),
        },
        "episodes": int(len(episodes)),
        "mean_length": _mean([float(item["length"]) for item in episodes]),
        "survival_rate": _mean([1.0 if item["length"] >= args.max_steps else 0.0 for item in episodes]),
        "mean_reward": _mean([float(item["reward"]) for item in episodes]),
        "mean_abs_vx_error": _mean([float(item["mean_abs_vx_error"]) for item in episodes]),
        "mean_abs_vz_error": _mean([float(item["mean_abs_vz_error"]) for item in episodes]),
        "mean_abs_yaw_error": _mean([float(item["mean_abs_yaw_error"]) for item in episodes]),
        "max_torque_limit_fraction": float(max((item["max_torque_limit_fraction"] for item in episodes), default=0.0)),
        "fraction_torque_saturated": _mean([float(item["fraction_torque_saturated"]) for item in episodes]),
        "total_contact_switches": int(sum(int(item["total_contact_switches"]) for item in episodes)),
        "reset_info": reset_infos,
    }
    for term in ACTIVE_REWARD_TERMS:
        summary[f"{term}_reward_mean"] = _mean([float(item[f"{term}_reward_mean"]) for item in episodes])
        summary[f"{term}_weighted_mean"] = _mean([float(item[f"{term}_weighted_mean"]) for item in episodes])

    (args.out / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    (args.out / "episodes.json").write_text(json.dumps(_json_ready(episodes), indent=2) + "\n", encoding="utf-8")
    if args.log_every_step:
        _write_csv(args.out / "timeline.csv", timeline)
    env.close()
    print(json.dumps(_json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
