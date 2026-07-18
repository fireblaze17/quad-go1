"""Headless tilt/contact diagnostics for trained Chrono Go1 policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from diagnostics import (
    FOOT_BODY_NAMES,
    LEG_PREFIXES,
    contact_body_groups,
    contact_debug_stats,
    foot_bodies,
    foot_debug_stats,
    foot_xz_positions,
)
from go1_env import Go1Env, _TIME_STEP
from project_config import DEFAULT_ACTION_FILTER_TAU, SB3_DEVICE


_LOAD_IMBALANCE_THRESHOLD = 0.25
_SLIP_THRESHOLD = 0.05
_ACTION_BIAS_THRESHOLD = 0.12
_JOINT_BIAS_THRESHOLD = 0.005
RESET_NOISE_LEVELS = ("clean", "rn1", "rn2", "rn3")
RESET_NOISE_COMPONENTS = ("combined", "joint_pos", "joint_vel", "roll_pitch", "base_height", "base_velocity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="Path to a Stable-Baselines3 policy zip.")
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.6)
    parser.add_argument("--friction-max", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.0)
    parser.add_argument("--ground-height-offset", type=float, default=0.0)
    parser.add_argument("--reset-noise-level", choices=RESET_NOISE_LEVELS, default="clean")
    parser.add_argument("--reset-noise-components", choices=RESET_NOISE_COMPONENTS, default="combined")
    parser.add_argument(
        "--tilt-threshold",
        type=float,
        default=0.003,
        help="First-step tilt_error threshold for tilt onset.",
    )
    parser.add_argument(
        "--unload-threshold",
        type=float,
        default=10.0,
        help="Foot vertical load threshold in N for first unload onset.",
    )
    parser.add_argument(
        "--log-every-step",
        "--timeline",
        dest="log_every_step",
        action="store_true",
        help="Write timeline.csv with one row per environment step.",
    )
    parser.add_argument(
        "--early-window-steps",
        type=int,
        default=250,
        help="Number of initial steps to summarize as the early recovery window.",
    )
    parser.add_argument(
        "--settled-window-steps",
        type=int,
        default=250,
        help="Number of final steps to summarize as the settled standing window.",
    )
    parser.add_argument(
        "--freeze-after-steps",
        type=int,
        default=None,
        help="After this many policy steps, reuse the mean action from the preceding freeze window.",
    )
    parser.add_argument(
        "--freeze-average-window",
        type=int,
        default=100,
        help="Number of policy actions to average before freezing.",
    )
    parser.add_argument(
        "--action-filter-tau",
        type=float,
        default=DEFAULT_ACTION_FILTER_TAU,
        help="Eval-only action low-pass filter time constant in seconds.",
    )
    args = parser.parse_args()
    if args.freeze_after_steps is not None and args.freeze_after_steps <= 0:
        parser.error("--freeze-after-steps must be positive when provided")
    if args.freeze_average_window <= 0:
        parser.error("--freeze-average-window must be positive")
    if (
        args.freeze_after_steps is not None
        and args.freeze_average_window > args.freeze_after_steps
    ):
        parser.error("--freeze-average-window cannot exceed --freeze-after-steps")
    if args.action_filter_tau is not None and args.action_filter_tau <= 0.0:
        parser.error("--action-filter-tau must be positive when provided")
    return args


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


def _foot_map(values: list[float]) -> dict[str, float]:
    return {
        name.split("_")[0]: float(value)
        for name, value in zip(FOOT_BODY_NAMES, values)
    }


def _term_foot_map(terms: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        leg: float(terms.get(f"{prefix}_{leg}", 0.0))
        for leg in LEG_PREFIXES
    }


def _leg_action_means(action: np.ndarray) -> dict[str, float]:
    return {
        leg: float(np.mean(np.abs(action[index * 3:(index + 1) * 3])))
        for index, leg in enumerate(LEG_PREFIXES)
    }


def _leg_action_rates(action_delta: np.ndarray) -> dict[str, float]:
    return {
        leg: float(np.mean(action_delta[index * 3:(index + 1) * 3] ** 2))
        for index, leg in enumerate(LEG_PREFIXES)
    }


def _leg_action_rate_means(action_deltas: np.ndarray) -> dict[str, float]:
    return {
        leg: float(np.mean(action_deltas[:, index * 3:(index + 1) * 3] ** 2))
        for index, leg in enumerate(LEG_PREFIXES)
    }


def _pair_sums(values: dict[str, float]) -> dict[str, float]:
    return {
        "front": values["FR"] + values["FL"],
        "rear": values["RR"] + values["RL"],
        "left": values["FL"] + values["RL"],
        "right": values["FR"] + values["RR"],
        "diag_fr_rl": values["FR"] + values["RL"],
        "diag_fl_rr": values["FL"] + values["RR"],
    }


def _max_pair_delta(pair_sums: dict[str, float]) -> float:
    return float(max(
        abs(pair_sums["front"] - pair_sums["rear"]),
        abs(pair_sums["left"] - pair_sums["right"]),
        abs(pair_sums["diag_fr_rl"] - pair_sums["diag_fl_rr"]),
    ))


def _yaw_from_obs(obs: np.ndarray) -> float:
    qw, qx, qy, qz = (float(value) for value in obs[1:5])
    forward_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    forward_z = 2.0 * (qx * qz - qw * qy)
    return float(np.arctan2(forward_z, forward_x))


def _angle_delta(current: float, reference: float) -> float:
    return float(np.arctan2(np.sin(current - reference), np.cos(current - reference)))


def _empty_window_summary() -> dict[str, Any]:
    return {
        "steps": 0,
        "mean_lin_vel_x": 0.0,
        "mean_lin_vel_z": 0.0,
        "mean_abs_xz_vel": 0.0,
        "mean_xz_speed": 0.0,
        "mean_base_world_y": 0.0,
        "mean_base_relative_height": 0.0,
        "min_base_relative_height": 0.0,
        "ground_top_y": 0.0,
        "ground_height_offset": 0.0,
        "base_displacement_from_reset": 0.0,
        "yaw_drift_from_reset": 0.0,
        "mean_trunk_x_up": 0.0,
        "mean_trunk_y_up": 0.0,
        "mean_abs_trunk_x_up": 0.0,
        "mean_abs_trunk_y_up": 0.0,
        "mean_tilt_error": 0.0,
        "min_foot_load": 0.0,
        "foot_load_shares": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_tangential_force_mean": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_tangential_force_max": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_friction_usage_mean": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_friction_usage_max": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_friction_usage_above_0p5_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_friction_usage_above_0p7_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_friction_usage_above_0p9_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_friction_usage_above_1p0_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "max_foot_friction_usage": 0.0,
        "worst_friction_usage_foot": "",
        "contact_duty": {leg: 0.0 for leg in LEG_PREFIXES},
        "contact_switches": {leg: 0 for leg in LEG_PREFIXES},
        "contact_foot_slip_distance": {leg: 0.0 for leg in LEG_PREFIXES},
        "total_contact_foot_slip_distance": 0.0,
        "foot_world_displacement": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_anchor_active_duty": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_anchor_displacement": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_anchor_error": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_anchor_reset_count": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_anchor_deactivate_count": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_anchor_off_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_load_below_20n_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_load_below_15n_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_load_below_8n_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "foot_load_below_5n_frames": {leg: 0.0 for leg in LEG_PREFIXES},
        "base_displacement_from_active_ref": 0.0,
        "action_mean_per_joint": [0.0] * 12,
        "action_std_per_joint": [0.0] * 12,
        "action_mean_abs_per_leg": {leg: 0.0 for leg in LEG_PREFIXES},
        "action_pair_delta": 0.0,
        "raw_action_mean_per_joint": [0.0] * 12,
        "raw_action_std_per_joint": [0.0] * 12,
        "raw_action_mean_abs_per_leg": {leg: 0.0 for leg in LEG_PREFIXES},
        "raw_action_pair_delta": 0.0,
        "executed_action_mean_per_joint": [0.0] * 12,
        "executed_action_std_per_joint": [0.0] * 12,
        "executed_action_mean_abs_per_leg": {leg: 0.0 for leg in LEG_PREFIXES},
        "executed_action_pair_delta": 0.0,
        "mean_abs_raw_action_delta": 0.0,
        "max_abs_raw_action_delta": 0.0,
        "raw_action_rate_per_leg": {leg: 0.0 for leg in LEG_PREFIXES},
        "mean_abs_executed_action_delta": 0.0,
        "max_abs_executed_action_delta": 0.0,
        "executed_action_rate_per_leg": {leg: 0.0 for leg in LEG_PREFIXES},
        "mean_filter_lag": 0.0,
        "max_filter_lag": 0.0,
        "joint_error_from_nominal": 0.0,
    }


def _window_summary(
    records: list[dict[str, Any]],
    reset_base_xz: tuple[float, float],
    reset_yaw: float,
    home_joint_angles: np.ndarray,
) -> dict[str, Any]:
    if not records:
        return _empty_window_summary()

    lin_x = np.array([item["lin_vel_x"] for item in records], dtype=np.float64)
    lin_z = np.array([item["lin_vel_z"] for item in records], dtype=np.float64)
    xz_speed = np.sqrt(lin_x ** 2 + lin_z ** 2)
    trunk_x_up = np.array([item["trunk_x_up"] for item in records], dtype=np.float64)
    trunk_y_up = np.array([item["trunk_y_up"] for item in records], dtype=np.float64)
    tilt_error = np.array([item["tilt_error"] for item in records], dtype=np.float64)
    base_world_y = np.array([item["base_world_y"] for item in records], dtype=np.float64)
    base_relative_height = np.array([item["base_relative_height"] for item in records], dtype=np.float64)
    actions = np.stack([item["action"] for item in records]).astype(np.float64)
    raw_actions = np.stack([item["raw_action"] for item in records]).astype(np.float64)
    executed_actions = np.stack([item["executed_action"] for item in records]).astype(np.float64)
    raw_action_deltas = np.stack([item["raw_action_delta"] for item in records]).astype(np.float64)
    executed_action_deltas = np.stack([item["executed_action_delta"] for item in records]).astype(np.float64)
    filter_lags = np.stack([item["filter_lag"] for item in records]).astype(np.float64)
    joints = np.stack([item["joint_pos"] for item in records]).astype(np.float64)
    final = records[-1]
    reset_base_x = float(final.get("base_reset_x", reset_base_xz[0]))
    reset_base_z = float(final.get("base_reset_z", reset_base_xz[1]))
    base_dx = float(final["base_x"] - reset_base_x)
    base_dz = float(final["base_z"] - reset_base_z)

    contact_duty = {}
    contact_switches = {}
    slip_distance = {}
    foot_displacement = {}
    anchor_active_duty = {}
    anchor_displacement = {}
    anchor_error = {}
    anchor_reset_count = {}
    anchor_deactivate_count = {}
    anchor_off_frames = {}
    below_count_maps = {
        "foot_load_below_20n_frames": {},
        "foot_load_below_15n_frames": {},
        "foot_load_below_8n_frames": {},
        "foot_load_below_5n_frames": {},
    }
    load_shares = {}
    tangential_force_mean = {}
    tangential_force_max = {}
    friction_usage_mean = {}
    friction_usage_max = {}
    usage_above_frames = {
        "foot_friction_usage_above_0p5_frames": {},
        "foot_friction_usage_above_0p7_frames": {},
        "foot_friction_usage_above_0p9_frames": {},
        "foot_friction_usage_above_1p0_frames": {},
    }
    min_foot_load = float("inf")
    for index, leg in enumerate(LEG_PREFIXES):
        contacts = [item["contacts"][index] for item in records]
        contact_duty[leg] = float(np.mean(contacts))
        contact_switches[leg] = int(sum(item["contact_switches"][index] for item in records))
        slip_distance[leg] = float(sum(item["contact_slip_increment"][index] for item in records))
        foot_displacement[leg] = float(final["foot_displacements"][index])
        anchor_active_duty[leg] = float(np.mean([item["foot_anchor_active"][leg] for item in records]))
        anchor_displacement[leg] = float(final["foot_anchor_displacement"][leg])
        anchor_error[leg] = float(final["foot_anchor_error"][leg])
        anchor_reset_count[leg] = float(final["foot_anchor_reset_count"][leg])
        anchor_deactivate_count[leg] = float(final["foot_anchor_deactivate_count"][leg])
        anchor_off_frames[leg] = float(final["foot_anchor_off_frames"][leg])
        for key in below_count_maps:
            below_count_maps[key][leg] = float(final[key][leg])
        load_shares[leg] = float(np.mean([item["foot_load_shares"][index] for item in records]))
        tangential_values = [item["foot_tangential_forces"][index] for item in records]
        usage_values = [item["foot_friction_usage"][index] for item in records]
        tangential_force_mean[leg] = float(np.mean(tangential_values))
        tangential_force_max[leg] = float(max(tangential_values, default=0.0))
        friction_usage_mean[leg] = float(np.mean(usage_values))
        friction_usage_max[leg] = float(max(usage_values, default=0.0))
        usage_above_frames["foot_friction_usage_above_0p5_frames"][leg] = float(
            sum(value > 0.5 for value in usage_values)
        )
        usage_above_frames["foot_friction_usage_above_0p7_frames"][leg] = float(
            sum(value > 0.7 for value in usage_values)
        )
        usage_above_frames["foot_friction_usage_above_0p9_frames"][leg] = float(
            sum(value > 0.9 for value in usage_values)
        )
        usage_above_frames["foot_friction_usage_above_1p0_frames"][leg] = float(
            sum(value > 1.0 for value in usage_values)
        )
        min_foot_load = min(min_foot_load, min(item["foot_loads"][index] for item in records))

    action_leg_means = _leg_action_means(np.mean(np.abs(actions), axis=0))
    raw_action_leg_means = _leg_action_means(np.mean(np.abs(raw_actions), axis=0))
    executed_action_leg_means = _leg_action_means(np.mean(np.abs(executed_actions), axis=0))
    raw_action_rate_per_leg = _leg_action_rate_means(raw_action_deltas)
    executed_action_rate_per_leg = _leg_action_rate_means(executed_action_deltas)
    active_ref_dx = float(final["base_x"] - final["base_ref_x"])
    active_ref_dz = float(final["base_z"] - final["base_ref_z"])
    return {
        "steps": len(records),
        "mean_lin_vel_x": float(np.mean(lin_x)),
        "mean_lin_vel_z": float(np.mean(lin_z)),
        "mean_abs_xz_vel": float(np.mean(np.abs(np.column_stack([lin_x, lin_z])))),
        "mean_xz_speed": float(np.mean(xz_speed)),
        "mean_base_world_y": float(np.mean(base_world_y)),
        "mean_base_relative_height": float(np.mean(base_relative_height)),
        "min_base_relative_height": float(np.min(base_relative_height)),
        "ground_top_y": float(final["ground_top_y"]),
        "ground_height_offset": float(final["ground_height_offset"]),
        "base_displacement_from_reset": float((base_dx * base_dx + base_dz * base_dz) ** 0.5),
        "base_dx_from_reset": base_dx,
        "base_dz_from_reset": base_dz,
        "base_ref_x": float(final["base_ref_x"]),
        "base_ref_z": float(final["base_ref_z"]),
        "base_displacement_from_active_ref": float((active_ref_dx * active_ref_dx + active_ref_dz * active_ref_dz) ** 0.5),
        "base_dx_from_active_ref": active_ref_dx,
        "base_dz_from_active_ref": active_ref_dz,
        "yaw_drift_from_reset": _angle_delta(float(final["yaw"]), reset_yaw),
        "mean_trunk_x_up": float(np.mean(trunk_x_up)),
        "mean_trunk_y_up": float(np.mean(trunk_y_up)),
        "mean_abs_trunk_x_up": float(np.mean(np.abs(trunk_x_up))),
        "mean_abs_trunk_y_up": float(np.mean(np.abs(trunk_y_up))),
        "mean_tilt_error": float(np.mean(tilt_error)),
        "min_foot_load": 0.0 if min_foot_load == float("inf") else float(min_foot_load),
        "foot_load_shares": load_shares,
        "foot_tangential_force_mean": tangential_force_mean,
        "foot_tangential_force_max": tangential_force_max,
        "foot_friction_usage_mean": friction_usage_mean,
        "foot_friction_usage_max": friction_usage_max,
        "max_foot_friction_usage": float(max(friction_usage_max.values(), default=0.0)),
        "worst_friction_usage_foot": max(friction_usage_max, key=friction_usage_max.get, default=""),
        **usage_above_frames,
        "contact_duty": contact_duty,
        "contact_switches": contact_switches,
        "total_contact_switches": int(sum(contact_switches.values())),
        "contact_foot_slip_distance": slip_distance,
        "total_contact_foot_slip_distance": float(sum(slip_distance.values())),
        "foot_world_displacement": foot_displacement,
        "max_foot_world_displacement": float(max(foot_displacement.values(), default=0.0)),
        "foot_anchor_active_duty": anchor_active_duty,
        "foot_anchor_displacement": anchor_displacement,
        "max_foot_anchor_displacement": float(max(anchor_displacement.values(), default=0.0)),
        "foot_anchor_error": anchor_error,
        "max_foot_anchor_error": float(max(anchor_error.values(), default=0.0)),
        "foot_anchor_reset_count": anchor_reset_count,
        "total_foot_anchor_resets": float(sum(anchor_reset_count.values())),
        "foot_anchor_deactivate_count": anchor_deactivate_count,
        "total_foot_anchor_deactivations": float(sum(anchor_deactivate_count.values())),
        "foot_anchor_off_frames": anchor_off_frames,
        **below_count_maps,
        "action_mean_per_joint": np.mean(actions, axis=0).astype(float).tolist(),
        "action_std_per_joint": np.std(actions, axis=0).astype(float).tolist(),
        "action_mean_abs_per_leg": action_leg_means,
        "action_pair_delta": _max_pair_delta(_pair_sums(action_leg_means)),
        "raw_action_mean_per_joint": np.mean(raw_actions, axis=0).astype(float).tolist(),
        "raw_action_std_per_joint": np.std(raw_actions, axis=0).astype(float).tolist(),
        "raw_action_mean_abs_per_leg": raw_action_leg_means,
        "raw_action_pair_delta": _max_pair_delta(_pair_sums(raw_action_leg_means)),
        "executed_action_mean_per_joint": np.mean(executed_actions, axis=0).astype(float).tolist(),
        "executed_action_std_per_joint": np.std(executed_actions, axis=0).astype(float).tolist(),
        "executed_action_mean_abs_per_leg": executed_action_leg_means,
        "executed_action_pair_delta": _max_pair_delta(_pair_sums(executed_action_leg_means)),
        "mean_abs_raw_action_delta": float(np.mean(np.abs(raw_action_deltas))),
        "max_abs_raw_action_delta": float(np.max(np.abs(raw_action_deltas))),
        "raw_action_rate_per_leg": raw_action_rate_per_leg,
        "mean_abs_executed_action_delta": float(np.mean(np.abs(executed_action_deltas))),
        "max_abs_executed_action_delta": float(np.max(np.abs(executed_action_deltas))),
        "executed_action_rate_per_leg": executed_action_rate_per_leg,
        "mean_filter_lag": float(np.mean(np.abs(filter_lags))),
        "max_filter_lag": float(np.max(np.abs(filter_lags))),
        "joint_error_from_nominal": float(np.mean((joints - home_joint_angles) ** 2)),
    }


def _classify_failure(settled: dict[str, Any], final_shares: dict[str, float]) -> str:
    drift = float(settled.get("base_displacement_from_reset", 0.0))
    xz_vel = float(settled.get("mean_abs_xz_vel", 0.0))
    slip = float(settled.get("total_contact_foot_slip_distance", 0.0))
    switches = int(settled.get("total_contact_switches", 0))
    action_bias = float(settled.get("action_pair_delta", 0.0))
    load_delta = max(final_shares.values(), default=0.25) - min(final_shares.values(), default=0.25)
    tilt = float(settled.get("mean_tilt_error", 0.0))

    moving = drift >= 0.03 or xz_vel >= 0.02
    if moving and slip >= 0.03:
        return "foot_slip"
    if moving and switches >= 2:
        return "micro_step"
    if moving or tilt >= 0.001 or load_delta >= 0.25:
        if action_bias >= _ACTION_BIAS_THRESHOLD or load_delta >= 0.20:
            return "planted_posture_bias"
        return "mixed"
    return "nominal"


def _dominant_load_pattern(shares: dict[str, float]) -> dict[str, Any]:
    pairs = _pair_sums(shares)
    groups = {
        "front_vs_rear": abs(pairs["front"] - pairs["rear"]),
        "left_vs_right": abs(pairs["left"] - pairs["right"]),
        "diag_vs_diag": abs(pairs["diag_fr_rl"] - pairs["diag_fl_rr"]),
    }
    dominant_axis = max(groups, key=groups.get)
    dominant_leg = max(shares, key=shares.get)
    unloaded_leg = min(shares, key=shares.get)
    return {
        "dominant_axis": dominant_axis,
        "dominant_axis_delta": float(groups[dominant_axis]),
        "dominant_leg": dominant_leg,
        "unloaded_leg": unloaded_leg,
        "pair_load_shares": pairs,
    }


def _tilt_direction(terms: dict[str, float]) -> str:
    x_up = float(terms.get("trunk_x_up", 0.0))
    y_up = float(terms.get("trunk_y_up", 0.0))
    parts = []
    if abs(x_up) >= 0.01:
        parts.append("trunk_x+_up" if x_up > 0.0 else "trunk_x-_up")
    if abs(y_up) >= 0.01:
        parts.append("trunk_y+_up" if y_up > 0.0 else "trunk_y-_up")
    return "/".join(parts) if parts else "near_upright"


def _event_step(event: dict[str, Any] | None) -> int | None:
    if event is None:
        return None
    return int(event["step"])


def _event_before(event: dict[str, Any] | None, reference: dict[str, Any] | None) -> bool:
    if event is None or reference is None:
        return False
    return int(event["step"]) <= int(reference["step"])


def _cause_hint(events: dict[str, Any]) -> str:
    tilt = events.get("first_tilt")
    if tilt is None:
        return "no_tilt_threshold_crossing"
    if _event_before(events.get("first_unload"), tilt):
        return "foot_unload_before_tilt"
    if _event_before(events.get("first_load_imbalance"), tilt):
        return "load_imbalance_before_tilt"
    if _event_before(events.get("first_slip"), tilt):
        return "foot_slip_before_tilt"
    if _event_before(events.get("first_action_bias"), tilt):
        return "action_bias_before_tilt"
    if _event_before(events.get("first_joint_bias"), tilt):
        return "joint_bias_before_tilt"
    return "tilt_before_recorded_precursor"


def _timeline_row(
    episode: int,
    step: int,
    friction: float,
    obs: np.ndarray,
    raw_action: np.ndarray,
    executed_action: np.ndarray,
    raw_action_delta: np.ndarray,
    executed_action_delta: np.ndarray,
    action_mode: str,
    action_filter_alpha: float | None,
    terms: dict[str, float],
    foot_stats: dict[str, Any],
    contact_stats: dict[str, Any],
) -> dict[str, Any]:
    foot_loads = _foot_map(foot_stats["foot_contact_loads"])
    foot_shares = _foot_map(foot_stats["foot_load_shares"])
    foot_speeds = _foot_map(foot_stats["foot_speeds"])
    foot_displacements = _foot_map(foot_stats["foot_displacements"])
    foot_tangential_forces = _foot_map(foot_stats["foot_tangential_forces"])
    foot_friction_usage = _foot_map(foot_stats["foot_friction_usage"])
    foot_positions = {
        name.split("_")[0]: [float(position[0]), float(position[1])]
        for name, position in zip(FOOT_BODY_NAMES, foot_stats["foot_positions"])
    }
    foot_anchor_refs = {
        leg: [
            float(terms.get(f"foot_anchor_ref_x_{leg}", 0.0)),
            float(terms.get(f"foot_anchor_ref_z_{leg}", 0.0)),
        ]
        for leg in LEG_PREFIXES
    }
    action_means = _leg_action_means(executed_action)
    raw_action_means = _leg_action_means(raw_action)
    executed_action_means = _leg_action_means(executed_action)
    raw_action_rates = _leg_action_rates(raw_action_delta)
    executed_action_rates = _leg_action_rates(executed_action_delta)
    return {
        "episode": episode,
        "step": step,
        "friction": friction,
        "base_x": float(terms.get("base_world_x", 0.0)),
        "base_z": float(terms.get("base_world_z", 0.0)),
        "base_ref_x": terms.get("base_ref_x", 0.0),
        "base_ref_z": terms.get("base_ref_z", 0.0),
        "base_reset_x": terms.get("base_reset_x", 0.0),
        "base_reset_z": terms.get("base_reset_z", 0.0),
        "base_drift": terms.get("base_drift", 0.0),
        "base_drift_from_reset": terms.get("base_drift_from_reset", 0.0),
        "standing_reference_captured": terms.get("standing_reference_captured", 0.0),
        "trunk_y": terms.get("trunk_y", 0.0),
        "base_world_y": terms.get("base_world_y", 0.0),
        "base_relative_height": terms.get("base_relative_height", 0.0),
        "ground_top_y": terms.get("ground_top_y", 0.0),
        "ground_height_offset": terms.get("ground_height_offset", 0.0),
        "upright_score": terms.get("upright_score", 0.0),
        "trunk_x_up": terms.get("trunk_x_up", 0.0),
        "trunk_y_up": terms.get("trunk_y_up", 0.0),
        "trunk_z_up": terms.get("trunk_z_up", 0.0),
        "tilt_error": terms.get("tilt_error", 0.0),
        "tilt_direction": _tilt_direction(terms),
        "reset_noise_enabled": terms.get("reset_noise_enabled", 0.0),
        "reset_noise_base_height_offset": terms.get("reset_noise_base_height_offset", 0.0),
        "reset_noise_roll": terms.get("reset_noise_roll", 0.0),
        "reset_noise_pitch": terms.get("reset_noise_pitch", 0.0),
        "reset_noise_joint_pos_offset_rms": terms.get("reset_noise_joint_pos_offset_rms", 0.0),
        "reset_noise_joint_vel_offset_rms": terms.get("reset_noise_joint_vel_offset_rms", 0.0),
        "reset_noise_base_linear_velocity_norm": terms.get("reset_noise_base_linear_velocity_norm", 0.0),
        "reset_noise_base_angular_velocity_norm": terms.get("reset_noise_base_angular_velocity_norm", 0.0),
        "reset_noise_initial_min_foot_y": terms.get("reset_noise_initial_min_foot_y", 0.0),
        "reset_noise_initial_max_foot_load": terms.get("reset_noise_initial_max_foot_load", 0.0),
        "lin_vel_x": terms.get("lin_vel_x", 0.0),
        "lin_vel_z": terms.get("lin_vel_z", 0.0),
        "mean_abs_action": terms.get("mean_abs_action", 0.0),
        "mean_abs_action_delta": terms.get("mean_abs_action_delta", 0.0),
        "action_mode": action_mode,
        "action_filter_alpha": "" if action_filter_alpha is None else action_filter_alpha,
        "mean_abs_raw_action_delta": float(np.mean(np.abs(raw_action_delta))),
        "max_abs_raw_action_delta": float(np.max(np.abs(raw_action_delta))),
        "mean_abs_executed_action_delta": float(np.mean(np.abs(executed_action_delta))),
        "max_abs_executed_action_delta": float(np.max(np.abs(executed_action_delta))),
        "mean_filter_lag": float(np.mean(np.abs(raw_action - executed_action))),
        "max_filter_lag": float(np.max(np.abs(raw_action - executed_action))),
        "mean_abs_joint_vel": terms.get("mean_abs_joint_vel", 0.0),
        "leg_symmetry_error": terms.get("leg_symmetry_error", 0.0),
        "foot_dxz_max": foot_stats["foot_dxz_max"],
        "foot_vxz_max": foot_stats["foot_vxz_max"],
        "foot_load_imbalance": foot_stats["foot_load_imbalance"],
        "max_foot_friction_usage": max(foot_stats["foot_friction_usage"], default=0.0),
        "foot_loads": json.dumps(foot_loads, sort_keys=True),
        "foot_shares": json.dumps(foot_shares, sort_keys=True),
        "foot_tangential_forces": json.dumps(foot_tangential_forces, sort_keys=True),
        "foot_friction_usage": json.dumps(foot_friction_usage, sort_keys=True),
        "foot_speeds": json.dumps(foot_speeds, sort_keys=True),
        "foot_displacements": json.dumps(foot_displacements, sort_keys=True),
        "foot_positions": json.dumps(foot_positions, sort_keys=True),
        "foot_anchor_active": json.dumps(_term_foot_map(terms, "foot_anchor_active"), sort_keys=True),
        "foot_anchor_refs": json.dumps(foot_anchor_refs, sort_keys=True),
        "foot_anchor_displacement": json.dumps(_term_foot_map(terms, "foot_anchor_displacement"), sort_keys=True),
        "foot_anchor_error": json.dumps(_term_foot_map(terms, "foot_anchor_error"), sort_keys=True),
        "foot_anchor_reset_count": json.dumps(_term_foot_map(terms, "foot_anchor_reset_count"), sort_keys=True),
        "foot_anchor_deactivate_count": json.dumps(_term_foot_map(terms, "foot_anchor_deactivate_count"), sort_keys=True),
        "foot_anchor_off_frames": json.dumps(_term_foot_map(terms, "foot_anchor_off_frames"), sort_keys=True),
        "foot_load_below_20n_frames": json.dumps(_term_foot_map(terms, "foot_load_below_20n_frames"), sort_keys=True),
        "foot_load_below_15n_frames": json.dumps(_term_foot_map(terms, "foot_load_below_15n_frames"), sort_keys=True),
        "foot_load_below_8n_frames": json.dumps(_term_foot_map(terms, "foot_load_below_8n_frames"), sort_keys=True),
        "foot_load_below_5n_frames": json.dumps(_term_foot_map(terms, "foot_load_below_5n_frames"), sort_keys=True),
        "foot_pair_shares": json.dumps(_pair_sums(foot_shares), sort_keys=True),
        "action_leg_means": json.dumps(action_means, sort_keys=True),
        "action_pair_means": json.dumps(_pair_sums(action_means), sort_keys=True),
        "raw_action_leg_means": json.dumps(raw_action_means, sort_keys=True),
        "executed_action_leg_means": json.dumps(executed_action_means, sort_keys=True),
        "raw_action_rate_per_leg": json.dumps(raw_action_rates, sort_keys=True),
        "executed_action_rate_per_leg": json.dumps(executed_action_rates, sort_keys=True),
        "nonfoot_loads": json.dumps(_foot_map(contact_stats["nonfoot_loads"]), sort_keys=True),
        "max_nonfoot_load": max(contact_stats["nonfoot_loads"]),
        "joint_positions": json.dumps(obs[11:23].astype(float).tolist()),
        "actions": json.dumps(np.asarray(executed_action, dtype=float).tolist()),
        "action_deltas": json.dumps(np.asarray(executed_action_delta, dtype=float).tolist()),
        "raw_actions": json.dumps(np.asarray(raw_action, dtype=float).tolist()),
        "executed_actions": json.dumps(np.asarray(executed_action, dtype=float).tolist()),
        "raw_action_deltas": json.dumps(np.asarray(raw_action_delta, dtype=float).tolist()),
        "executed_action_deltas": json.dumps(np.asarray(executed_action_delta, dtype=float).tolist()),
        "filter_lag": json.dumps(np.asarray(raw_action - executed_action, dtype=float).tolist()),
    }


def _update_first_events(
    events: dict[str, Any],
    step: int,
    terms: dict[str, float],
    foot_stats: dict[str, Any],
    contact_stats: dict[str, Any],
    action: np.ndarray,
    tilt_threshold: float,
    unload_threshold: float,
) -> None:
    foot_loads = _foot_map(foot_stats["foot_contact_loads"])
    foot_shares = _foot_map(foot_stats["foot_load_shares"])
    action_means = _leg_action_means(action)
    action_pair_delta = _max_pair_delta(_pair_sums(action_means))

    if events["first_tilt"] is None and float(terms.get("tilt_error", 0.0)) >= tilt_threshold:
        events["first_tilt"] = {
            "step": step,
            "tilt_error": float(terms.get("tilt_error", 0.0)),
            "direction": _tilt_direction(terms),
        }

    if events["first_unload"] is None and min(foot_loads.values()) <= unload_threshold:
        unloaded_leg = min(foot_loads, key=foot_loads.get)
        events["first_unload"] = {
            "step": step,
            "leg": unloaded_leg,
            "load": float(foot_loads[unloaded_leg]),
        }

    if (
        events["first_load_imbalance"] is None
        and float(foot_stats["foot_load_imbalance"]) >= _LOAD_IMBALANCE_THRESHOLD
    ):
        events["first_load_imbalance"] = {
            "step": step,
            "load_imbalance": float(foot_stats["foot_load_imbalance"]),
            **_dominant_load_pattern(foot_shares),
        }

    if events["first_slip"] is None and float(foot_stats["foot_dxz_max"]) >= _SLIP_THRESHOLD:
        events["first_slip"] = {
            "step": step,
            "foot_dxz_max": float(foot_stats["foot_dxz_max"]),
            "foot_vxz_max": float(foot_stats["foot_vxz_max"]),
        }

    if events["first_action_bias"] is None and action_pair_delta >= _ACTION_BIAS_THRESHOLD:
        events["first_action_bias"] = {
            "step": step,
            "pair_delta": action_pair_delta,
            "action_pair_means": _pair_sums(action_means),
        }

    if (
        events["first_joint_bias"] is None
        and float(terms.get("leg_symmetry_error", 0.0)) >= _JOINT_BIAS_THRESHOLD
    ):
        events["first_joint_bias"] = {
            "step": step,
            "leg_symmetry_error": float(terms.get("leg_symmetry_error", 0.0)),
        }

    if events["first_nonfoot_load"] is None and max(contact_stats["nonfoot_loads"]) > 1e-6:
        events["first_nonfoot_load"] = {
            "step": step,
            "nonfoot_loads": _foot_map(contact_stats["nonfoot_loads"]),
        }


def _write_timeline_header(writer: csv.DictWriter, row: dict[str, Any], state: dict[str, bool]) -> None:
    if not state["wrote_header"]:
        writer.fieldnames = list(row)
        writer.writeheader()
        state["wrote_header"] = True


def diagnose_episode(
    env: Go1Env,
    model: PPO,
    episode: int,
    args,
    timeline_writer: csv.DictWriter | None,
    timeline_state: dict[str, bool],
) -> dict[str, Any]:
    obs, info = env.reset()
    friction = float(info["ground_friction"])
    spawn = {
        "x": float(info.get("spawn_x", args.spawn_x)),
        "z": float(info.get("spawn_z", args.spawn_z)),
    }
    material_properties = info.get("material_properties", {})
    reset_noise = info.get("reset_noise", {})
    effective_friction = material_properties.get("effective_friction", info.get("effective_friction", friction))
    effective_friction = None if effective_friction is None else float(effective_friction)
    tracked_feet = foot_bodies(env)
    tracked_contacts = contact_body_groups(env)
    reset_foot_xz = foot_xz_positions(tracked_feet)
    reset_base_xz = (float(spawn["x"]), float(spawn["z"]))
    reset_yaw = _yaw_from_obs(obs)
    prev_raw_action = np.zeros(12, dtype=np.float32)
    prev_executed_action = np.zeros(12, dtype=np.float32)
    filter_initialized = False
    action_filter_alpha = None
    if args.action_filter_tau is not None:
        action_filter_alpha = float(_TIME_STEP / (args.action_filter_tau + _TIME_STEP))
    prev_contacts = [False] * len(FOOT_BODY_NAMES)
    freeze_action_buffer: list[np.ndarray] = []
    frozen_action: np.ndarray | None = None
    records: list[dict[str, Any]] = []

    events = {
        "first_tilt": None,
        "first_unload": None,
        "first_load_imbalance": None,
        "first_slip": None,
        "first_action_bias": None,
        "first_joint_bias": None,
        "first_nonfoot_load": None,
    }
    max_values = {
        "max_tilt_error": 0.0,
        "max_load_imbalance": 0.0,
        "max_foot_dxz": 0.0,
        "max_foot_vxz": 0.0,
        "max_nonfoot_load": 0.0,
        "max_trunk_stance_center_error": 0.0,
        "max_foot_anchor_displacement": 0.0,
        "max_foot_anchor_error": 0.0,
        "max_foot_anchor_resets": 0.0,
        "max_foot_anchor_deactivations": 0.0,
        "max_foot_friction_usage": 0.0,
        "min_foot_load": float("inf"),
        "min_upright_score": float("inf"),
    }
    totals = {
        "tilt_error": 0.0,
        "load_imbalance": 0.0,
        "foot_dxz_max": 0.0,
        "trunk_stance_center_error": 0.0,
        "trunk_stance_center_dx": 0.0,
        "trunk_stance_center_dz": 0.0,
        "mean_abs_action": 0.0,
        "leg_symmetry_error": 0.0,
    }

    steps = 0
    total_reward = 0.0
    done = False
    terminated = False
    truncated = False
    final_terms: dict[str, float] = {}
    final_foot_stats: dict[str, Any] = {}

    while not done:
        if frozen_action is None:
            raw_action, _ = model.predict(obs, deterministic=True)
            raw_action = np.asarray(raw_action, dtype=np.float32)
            action_mode = "policy"
        else:
            raw_action = frozen_action.copy()
            action_mode = "frozen"
        if action_filter_alpha is None:
            executed_action = raw_action.copy()
        elif not filter_initialized:
            executed_action = raw_action.copy()
            filter_initialized = True
        else:
            executed_action = prev_executed_action + action_filter_alpha * (raw_action - prev_executed_action)
            executed_action = executed_action.astype(np.float32)
        raw_action_delta = raw_action - prev_raw_action
        executed_action_delta = executed_action - prev_executed_action
        obs, reward, terminated, truncated, info = env.step(executed_action)
        steps += 1
        if action_mode == "policy":
            freeze_action_buffer.append(raw_action.copy())
            if len(freeze_action_buffer) > args.freeze_average_window:
                freeze_action_buffer.pop(0)
            if args.freeze_after_steps is not None and steps == args.freeze_after_steps:
                frozen_action = np.mean(np.stack(freeze_action_buffer), axis=0).astype(np.float32)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        terms = info.get("reward_terms", {})
        foot_stats = foot_debug_stats(tracked_feet, reset_foot_xz, effective_friction)
        contact_stats = contact_debug_stats(tracked_contacts)
        foot_loads = foot_stats["foot_contact_loads"]
        contacts = [float(load) > args.unload_threshold for load in foot_loads]
        contact_switches = [
            int(step_contact != prev_contact)
            for step_contact, prev_contact in zip(contacts, prev_contacts)
        ]
        contact_slip_increment = [
            float(speed) * _TIME_STEP if contact else 0.0
            for speed, contact in zip(foot_stats["foot_speeds"], contacts)
        ]

        _update_first_events(
            events,
            steps,
            terms,
            foot_stats,
            contact_stats,
            executed_action,
            args.tilt_threshold,
            args.unload_threshold,
        )

        max_values["max_tilt_error"] = max(max_values["max_tilt_error"], float(terms.get("tilt_error", 0.0)))
        max_values["max_load_imbalance"] = max(max_values["max_load_imbalance"], float(foot_stats["foot_load_imbalance"]))
        max_values["max_foot_dxz"] = max(max_values["max_foot_dxz"], float(foot_stats["foot_dxz_max"]))
        max_values["max_foot_vxz"] = max(max_values["max_foot_vxz"], float(foot_stats["foot_vxz_max"]))
        max_values["max_nonfoot_load"] = max(max_values["max_nonfoot_load"], float(max(contact_stats["nonfoot_loads"])))
        max_values["max_trunk_stance_center_error"] = max(
            max_values["max_trunk_stance_center_error"],
            float(terms.get("trunk_stance_center_error", 0.0)),
        )
        max_values["max_foot_anchor_displacement"] = max(
            max_values["max_foot_anchor_displacement"],
            float(terms.get("foot_anchor_max_displacement", 0.0)),
        )
        max_values["max_foot_anchor_error"] = max(
            max_values["max_foot_anchor_error"],
            max(_term_foot_map(terms, "foot_anchor_error").values(), default=0.0),
        )
        max_values["max_foot_anchor_resets"] = max(
            max_values["max_foot_anchor_resets"],
            float(terms.get("foot_anchor_total_resets", 0.0)),
        )
        max_values["max_foot_anchor_deactivations"] = max(
            max_values["max_foot_anchor_deactivations"],
            float(terms.get("foot_anchor_total_deactivations", 0.0)),
        )
        max_values["max_foot_friction_usage"] = max(
            max_values["max_foot_friction_usage"],
            float(max(foot_stats["foot_friction_usage"], default=0.0)),
        )
        max_values["min_foot_load"] = min(max_values["min_foot_load"], float(min(foot_loads)))
        max_values["min_upright_score"] = min(
            max_values["min_upright_score"],
            float(terms.get("upright_score", float("inf"))),
        )
        totals["tilt_error"] += float(terms.get("tilt_error", 0.0))
        totals["load_imbalance"] += float(foot_stats["foot_load_imbalance"])
        totals["foot_dxz_max"] += float(foot_stats["foot_dxz_max"])
        totals["trunk_stance_center_error"] += float(terms.get("trunk_stance_center_error", 0.0))
        totals["trunk_stance_center_dx"] += float(terms.get("trunk_stance_center_dx", 0.0))
        totals["trunk_stance_center_dz"] += float(terms.get("trunk_stance_center_dz", 0.0))
        totals["mean_abs_action"] += float(terms.get("mean_abs_action", 0.0))
        totals["leg_symmetry_error"] += float(terms.get("leg_symmetry_error", 0.0))

        records.append({
            "step": steps,
            "base_x": float(terms.get("base_world_x", 0.0)),
            "base_z": float(terms.get("base_world_z", 0.0)),
            "base_world_y": float(terms.get("base_world_y", 0.0)),
            "base_relative_height": float(terms.get("base_relative_height", 0.0)),
            "ground_top_y": float(terms.get("ground_top_y", 0.0)),
            "ground_height_offset": float(terms.get("ground_height_offset", 0.0)),
            "base_reset_x": float(terms.get("base_reset_x", 0.0)),
            "base_reset_z": float(terms.get("base_reset_z", 0.0)),
            "base_ref_x": float(terms.get("base_ref_x", 0.0)),
            "base_ref_z": float(terms.get("base_ref_z", 0.0)),
            "yaw": _yaw_from_obs(obs),
            "lin_vel_x": float(terms.get("lin_vel_x", 0.0)),
            "lin_vel_z": float(terms.get("lin_vel_z", 0.0)),
            "trunk_x_up": float(terms.get("trunk_x_up", 0.0)),
            "trunk_y_up": float(terms.get("trunk_y_up", 0.0)),
            "tilt_error": float(terms.get("tilt_error", 0.0)),
            "foot_loads": [float(value) for value in foot_stats["foot_contact_loads"]],
            "foot_load_shares": [float(value) for value in foot_stats["foot_load_shares"]],
            "foot_tangential_forces": [float(value) for value in foot_stats["foot_tangential_forces"]],
            "foot_friction_usage": [float(value) for value in foot_stats["foot_friction_usage"]],
            "foot_displacements": [float(value) for value in foot_stats["foot_displacements"]],
            "foot_anchor_active": _term_foot_map(terms, "foot_anchor_active"),
            "foot_anchor_displacement": _term_foot_map(terms, "foot_anchor_displacement"),
            "foot_anchor_error": _term_foot_map(terms, "foot_anchor_error"),
            "foot_anchor_reset_count": _term_foot_map(terms, "foot_anchor_reset_count"),
            "foot_anchor_deactivate_count": _term_foot_map(terms, "foot_anchor_deactivate_count"),
            "foot_anchor_off_frames": _term_foot_map(terms, "foot_anchor_off_frames"),
            "foot_load_below_20n_frames": _term_foot_map(terms, "foot_load_below_20n_frames"),
            "foot_load_below_15n_frames": _term_foot_map(terms, "foot_load_below_15n_frames"),
            "foot_load_below_8n_frames": _term_foot_map(terms, "foot_load_below_8n_frames"),
            "foot_load_below_5n_frames": _term_foot_map(terms, "foot_load_below_5n_frames"),
            "contacts": contacts,
            "contact_switches": contact_switches,
            "contact_slip_increment": contact_slip_increment,
            "action_mode": action_mode,
            "action": executed_action.astype(float).copy(),
            "raw_action": raw_action.astype(float).copy(),
            "executed_action": executed_action.astype(float).copy(),
            "raw_action_delta": raw_action_delta.astype(float).copy(),
            "executed_action_delta": executed_action_delta.astype(float).copy(),
            "filter_lag": (raw_action - executed_action).astype(float).copy(),
            "joint_pos": obs[11:23].astype(float).copy(),
        })

        if timeline_writer is not None:
            row = _timeline_row(
                episode,
                steps,
                friction,
                obs,
                raw_action,
                executed_action,
                raw_action_delta,
                executed_action_delta,
                action_mode,
                action_filter_alpha,
                terms,
                foot_stats,
                contact_stats,
            )
            _write_timeline_header(timeline_writer, row, timeline_state)
            timeline_writer.writerow(row)

        prev_raw_action = raw_action.copy()
        prev_executed_action = executed_action.copy()
        prev_contacts = contacts
        final_terms = terms
        final_foot_stats = foot_stats

    final_shares = _foot_map(final_foot_stats["foot_load_shares"])
    early_records = records[:max(0, args.early_window_steps)]
    settled_records = records[-max(0, args.settled_window_steps):] if args.settled_window_steps > 0 else []
    windows = {
        "full_episode": _window_summary(records, reset_base_xz, reset_yaw, env.home_joint_angles),
        "early_window": _window_summary(early_records, reset_base_xz, reset_yaw, env.home_joint_angles),
        "settled_window": _window_summary(settled_records, reset_base_xz, reset_yaw, env.home_joint_angles),
    }
    settled = windows["settled_window"]
    failure_type = _classify_failure(settled, final_shares)
    summary = {
        "episode": episode,
        "steps": steps,
        "reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": info.get("termination_reason") or ("truncated" if truncated else "unknown"),
        "friction": friction,
        "effective_friction": effective_friction,
        "spawn": spawn,
        "material_properties": material_properties,
        "reset_noise": reset_noise,
        "freeze_action": {
            "enabled": args.freeze_after_steps is not None,
            "freeze_step": args.freeze_after_steps,
            "freeze_average_window": args.freeze_average_window,
            "frozen_action": None if frozen_action is None else frozen_action.astype(float).tolist(),
        },
        "action_filter": {
            "enabled": args.action_filter_tau is not None,
            "tau": args.action_filter_tau,
            "alpha": action_filter_alpha,
        },
        "final_tilt_direction": _tilt_direction(final_terms),
        "final_axis_up": {
            "trunk_x_up": float(final_terms.get("trunk_x_up", 0.0)),
            "trunk_y_up": float(final_terms.get("trunk_y_up", 0.0)),
            "trunk_z_up": float(final_terms.get("trunk_z_up", 0.0)),
        },
        "final_load_pattern": _dominant_load_pattern(final_shares),
        "events": events,
        "cause_hint": _cause_hint(events),
        "failure_type": failure_type,
        "event_steps": {key: _event_step(value) for key, value in events.items()},
        "max_values": max_values,
        "means": {
            key: value / max(1, steps)
            for key, value in totals.items()
        },
        "windows": windows,
    }
    return _json_ready(summary)


def _mean_dict_values(items: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, float]:
    return {
        key: float(np.mean([item.get(key, 0.0) for item in items])) if items else 0.0
        for key in keys
    }


def _aggregate_window(episodes: list[dict[str, Any]], window_name: str) -> dict[str, Any]:
    windows = [item["windows"][window_name] for item in episodes if window_name in item.get("windows", {})]
    if not windows:
        return _empty_window_summary()
    aggregate_friction_usage_max = _mean_dict_values(
        [window["foot_friction_usage_max"] for window in windows],
        LEG_PREFIXES,
    )
    scalar_keys = (
        "mean_lin_vel_x",
        "mean_lin_vel_z",
        "mean_abs_xz_vel",
        "mean_xz_speed",
        "base_displacement_from_reset",
        "base_dx_from_reset",
        "base_dz_from_reset",
        "base_displacement_from_active_ref",
        "base_dx_from_active_ref",
        "base_dz_from_active_ref",
        "yaw_drift_from_reset",
        "mean_trunk_x_up",
        "mean_trunk_y_up",
        "mean_abs_trunk_x_up",
        "mean_abs_trunk_y_up",
        "mean_tilt_error",
        "min_foot_load",
        "max_foot_friction_usage",
        "total_contact_switches",
        "total_contact_foot_slip_distance",
        "max_foot_world_displacement",
        "max_foot_anchor_displacement",
        "max_foot_anchor_error",
        "total_foot_anchor_resets",
        "total_foot_anchor_deactivations",
        "action_pair_delta",
        "raw_action_pair_delta",
        "executed_action_pair_delta",
        "mean_abs_raw_action_delta",
        "max_abs_raw_action_delta",
        "mean_abs_executed_action_delta",
        "max_abs_executed_action_delta",
        "mean_filter_lag",
        "max_filter_lag",
        "joint_error_from_nominal",
    )
    result = {
        "steps": int(np.mean([window["steps"] for window in windows])),
        **{
            key: float(np.mean([window.get(key, 0.0) for window in windows]))
            for key in scalar_keys
        },
        "foot_load_shares": _mean_dict_values(
            [window["foot_load_shares"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_tangential_force_mean": _mean_dict_values(
            [window["foot_tangential_force_mean"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_tangential_force_max": _mean_dict_values(
            [window["foot_tangential_force_max"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_friction_usage_mean": _mean_dict_values(
            [window["foot_friction_usage_mean"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_friction_usage_max": _mean_dict_values(
            [window["foot_friction_usage_max"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_friction_usage_above_0p5_frames": _mean_dict_values(
            [window["foot_friction_usage_above_0p5_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_friction_usage_above_0p7_frames": _mean_dict_values(
            [window["foot_friction_usage_above_0p7_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_friction_usage_above_0p9_frames": _mean_dict_values(
            [window["foot_friction_usage_above_0p9_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_friction_usage_above_1p0_frames": _mean_dict_values(
            [window["foot_friction_usage_above_1p0_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "worst_friction_usage_foot": max(
            aggregate_friction_usage_max,
            key=aggregate_friction_usage_max.get,
            default="",
        ),
        "contact_duty": _mean_dict_values(
            [window["contact_duty"] for window in windows],
            LEG_PREFIXES,
        ),
        "contact_switches": _mean_dict_values(
            [window["contact_switches"] for window in windows],
            LEG_PREFIXES,
        ),
        "contact_foot_slip_distance": _mean_dict_values(
            [window["contact_foot_slip_distance"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_world_displacement": _mean_dict_values(
            [window["foot_world_displacement"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_anchor_active_duty": _mean_dict_values(
            [window["foot_anchor_active_duty"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_anchor_displacement": _mean_dict_values(
            [window["foot_anchor_displacement"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_anchor_error": _mean_dict_values(
            [window["foot_anchor_error"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_anchor_reset_count": _mean_dict_values(
            [window["foot_anchor_reset_count"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_anchor_deactivate_count": _mean_dict_values(
            [window["foot_anchor_deactivate_count"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_anchor_off_frames": _mean_dict_values(
            [window["foot_anchor_off_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_load_below_20n_frames": _mean_dict_values(
            [window["foot_load_below_20n_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_load_below_15n_frames": _mean_dict_values(
            [window["foot_load_below_15n_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_load_below_8n_frames": _mean_dict_values(
            [window["foot_load_below_8n_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "foot_load_below_5n_frames": _mean_dict_values(
            [window["foot_load_below_5n_frames"] for window in windows],
            LEG_PREFIXES,
        ),
        "action_mean_abs_per_leg": _mean_dict_values(
            [window["action_mean_abs_per_leg"] for window in windows],
            LEG_PREFIXES,
        ),
        "raw_action_mean_abs_per_leg": _mean_dict_values(
            [window["raw_action_mean_abs_per_leg"] for window in windows],
            LEG_PREFIXES,
        ),
        "executed_action_mean_abs_per_leg": _mean_dict_values(
            [window["executed_action_mean_abs_per_leg"] for window in windows],
            LEG_PREFIXES,
        ),
        "raw_action_rate_per_leg": _mean_dict_values(
            [window["raw_action_rate_per_leg"] for window in windows],
            LEG_PREFIXES,
        ),
        "executed_action_rate_per_leg": _mean_dict_values(
            [window["executed_action_rate_per_leg"] for window in windows],
            LEG_PREFIXES,
        ),
        "p95": {
            key: float(np.percentile([window.get(key, 0.0) for window in windows], 95))
            for key in (
                "mean_abs_xz_vel",
                "base_displacement_from_reset",
                "base_displacement_from_active_ref",
                "mean_tilt_error",
                "total_contact_foot_slip_distance",
                "total_contact_switches",
                "max_foot_anchor_displacement",
                "max_foot_friction_usage",
                "total_foot_anchor_deactivations",
                "mean_abs_executed_action_delta",
                "max_abs_executed_action_delta",
            )
        },
    }
    return result


def _episode_replay_metadata(item: dict[str, Any], metric: str, value: float) -> dict[str, Any]:
    settled = item.get("windows", {}).get("settled_window", {})
    pattern = item["final_load_pattern"]
    return {
        "episode": item["episode"],
        "metric": metric,
        "value": float(value),
        "friction": item["friction"],
        "dominant_leg": pattern["dominant_leg"],
        "least_loaded_leg": pattern["unloaded_leg"],
        "failure_type": item.get("failure_type"),
        "final_tilt_direction": item.get("final_tilt_direction"),
        "settled_summary": {
            "mean_abs_xz_vel": settled.get("mean_abs_xz_vel"),
            "base_displacement_from_reset": settled.get("base_displacement_from_reset"),
            "base_displacement_from_active_ref": settled.get("base_displacement_from_active_ref"),
            "mean_tilt_error": settled.get("mean_tilt_error"),
            "total_contact_foot_slip_distance": settled.get("total_contact_foot_slip_distance"),
            "total_contact_switches": settled.get("total_contact_switches"),
            "min_foot_load": settled.get("min_foot_load"),
            "foot_load_shares": settled.get("foot_load_shares"),
            "foot_friction_usage_mean": settled.get("foot_friction_usage_mean"),
            "foot_friction_usage_max": settled.get("foot_friction_usage_max"),
            "max_foot_friction_usage": settled.get("max_foot_friction_usage"),
            "worst_friction_usage_foot": settled.get("worst_friction_usage_foot"),
            "foot_anchor_active_duty": settled.get("foot_anchor_active_duty"),
            "foot_anchor_reset_count": settled.get("foot_anchor_reset_count"),
            "foot_anchor_deactivate_count": settled.get("foot_anchor_deactivate_count"),
            "foot_load_below_20n_frames": settled.get("foot_load_below_20n_frames"),
            "foot_load_below_5n_frames": settled.get("foot_load_below_5n_frames"),
        },
    }


def _top_worst_episodes(episodes: list[dict[str, Any]], top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
    metrics = {
        "settled_drift": lambda item: item["windows"]["settled_window"].get("base_displacement_from_reset", 0.0),
        "settled_tilt": lambda item: item["windows"]["settled_window"].get("mean_tilt_error", 0.0),
        "settled_load_imbalance": lambda item: (
            max(item["windows"]["settled_window"].get("foot_load_shares", {}).values(), default=0.0)
            - min(item["windows"]["settled_window"].get("foot_load_shares", {}).values(), default=0.0)
        ),
        "settled_slip": lambda item: item["windows"]["settled_window"].get("total_contact_foot_slip_distance", 0.0),
    }
    worst = {}
    for name, metric_fn in metrics.items():
        ranked = sorted(episodes, key=metric_fn, reverse=True)[:top_k]
        worst[name] = [
            _episode_replay_metadata(item, name, metric_fn(item))
            for item in ranked
        ]
    return worst


def aggregate_summaries(episodes: list[dict[str, Any]], args) -> dict[str, Any]:
    failures = sum(1 for item in episodes if item["terminated"])
    cause_counts: dict[str, int] = {}
    failure_type_counts: dict[str, int] = {}
    tilt_directions: dict[str, int] = {}
    dominant_axes: dict[str, int] = {}
    dominant_legs: dict[str, int] = {}
    unloaded_legs: dict[str, int] = {}

    for item in episodes:
        cause_counts[item["cause_hint"]] = cause_counts.get(item["cause_hint"], 0) + 1
        failure_type = item.get("failure_type", "unknown")
        failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
        tilt_directions[item["final_tilt_direction"]] = (
            tilt_directions.get(item["final_tilt_direction"], 0) + 1
        )
        pattern = item["final_load_pattern"]
        dominant_axes[pattern["dominant_axis"]] = dominant_axes.get(pattern["dominant_axis"], 0) + 1
        dominant_legs[pattern["dominant_leg"]] = dominant_legs.get(pattern["dominant_leg"], 0) + 1
        unloaded_legs[pattern["unloaded_leg"]] = unloaded_legs.get(pattern["unloaded_leg"], 0) + 1

    lengths = [item["steps"] for item in episodes]
    rewards = [item["reward"] for item in episodes]
    max_values = [item["max_values"] for item in episodes]
    material_properties = episodes[0].get("material_properties", {}) if episodes else {}
    effective_frictions = [
        item["effective_friction"]
        for item in episodes
        if item.get("effective_friction") is not None
    ]
    return _json_ready({
        "policy": str(args.policy),
        "terrain": args.terrain,
        "friction_range": [args.friction_min, args.friction_max],
        "spawn": {
            "x": args.spawn_x,
            "z": args.spawn_z,
        },
        "ground_height_offset": args.ground_height_offset,
        "material_properties": material_properties,
        "reset_noise": {
            "level": args.reset_noise_level,
            "components": args.reset_noise_components,
            "episode_samples": [item.get("reset_noise", {}) for item in episodes],
        },
        "freeze_action": {
            "enabled": args.freeze_after_steps is not None,
            "freeze_step": args.freeze_after_steps,
            "freeze_average_window": args.freeze_average_window,
            "frozen_actions": [
                item.get("freeze_action", {}).get("frozen_action")
                for item in episodes
            ],
        },
        "action_filter": {
            "enabled": args.action_filter_tau is not None,
            "tau": args.action_filter_tau,
            "alpha": None if args.action_filter_tau is None else float(_TIME_STEP / (args.action_filter_tau + _TIME_STEP)),
        },
        "episodes": len(episodes),
        "survival_rate": 1.0 - failures / max(1, len(episodes)),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "friction_min_seen": min((item["friction"] for item in episodes), default=None),
        "friction_max_seen": max((item["friction"] for item in episodes), default=None),
        "effective_friction_min_seen": min(effective_frictions, default=None),
        "effective_friction_max_seen": max(effective_frictions, default=None),
        "cause_counts": cause_counts,
        "failure_type_counts": failure_type_counts,
        "final_tilt_directions": tilt_directions,
        "dominant_load_axes": dominant_axes,
        "dominant_loaded_legs": dominant_legs,
        "least_loaded_legs": unloaded_legs,
        "worst_case": {
            "max_tilt_error": max((item["max_tilt_error"] for item in max_values), default=0.0),
            "max_load_imbalance": max((item["max_load_imbalance"] for item in max_values), default=0.0),
            "max_foot_dxz": max((item["max_foot_dxz"] for item in max_values), default=0.0),
            "max_trunk_stance_center_error": max(
                (item["max_trunk_stance_center_error"] for item in max_values),
                default=0.0,
            ),
            "max_foot_anchor_displacement": max(
                (item["max_foot_anchor_displacement"] for item in max_values),
                default=0.0,
            ),
            "max_foot_anchor_error": max((item["max_foot_anchor_error"] for item in max_values), default=0.0),
            "max_foot_anchor_resets": max((item["max_foot_anchor_resets"] for item in max_values), default=0.0),
            "max_foot_anchor_deactivations": max(
                (item["max_foot_anchor_deactivations"] for item in max_values),
                default=0.0,
            ),
            "max_foot_friction_usage": max(
                (item["max_foot_friction_usage"] for item in max_values),
                default=0.0,
            ),
            "max_nonfoot_load": max((item["max_nonfoot_load"] for item in max_values), default=0.0),
            "min_foot_load": min((item["min_foot_load"] for item in max_values), default=0.0),
            "min_upright_score": min((item["min_upright_score"] for item in max_values), default=0.0),
        },
        "window_means": {
            "full_episode": _aggregate_window(episodes, "full_episode"),
            "early_window": _aggregate_window(episodes, "early_window"),
            "settled_window": _aggregate_window(episodes, "settled_window"),
        },
        "worst_episodes": _top_worst_episodes(episodes),
        "thresholds": {
            "tilt_error": args.tilt_threshold,
            "unload_n": args.unload_threshold,
            "load_imbalance": _LOAD_IMBALANCE_THRESHOLD,
            "slip_m": _SLIP_THRESHOLD,
            "action_pair_delta": _ACTION_BIAS_THRESHOLD,
            "joint_symmetry_error": _JOINT_BIAS_THRESHOLD,
            "early_window_steps": args.early_window_steps,
            "settled_window_steps": args.settled_window_steps,
        },
    })


def print_aggregate(aggregate: dict[str, Any]) -> None:
    print(f"policy: {aggregate['policy']}")
    print(f"episodes: {aggregate['episodes']}")
    print(f"survival_rate: {aggregate['survival_rate']:.3f}")
    print(f"mean_length: {aggregate['mean_length']:.1f}")
    print(f"mean_reward: {aggregate['mean_reward']:.3f}")
    print(f"friction_min_seen: {aggregate['friction_min_seen']:.3f}")
    print(f"friction_max_seen: {aggregate['friction_max_seen']:.3f}")
    if aggregate.get("effective_friction_min_seen") is not None:
        print(f"effective_friction_min_seen: {aggregate['effective_friction_min_seen']:.3f}")
        print(f"effective_friction_max_seen: {aggregate['effective_friction_max_seen']:.3f}")
    if aggregate.get("freeze_action", {}).get("enabled"):
        freeze = aggregate["freeze_action"]
        print(
            "freeze_action: "
            f"step={freeze['freeze_step']} "
            f"window={freeze['freeze_average_window']}"
        )
    if aggregate.get("action_filter", {}).get("enabled"):
        action_filter = aggregate["action_filter"]
        print(
            "action_filter: "
            f"tau={action_filter['tau']} "
            f"alpha={action_filter['alpha']:.6f}"
        )
    print(f"cause_counts: {aggregate['cause_counts']}")
    print(f"failure_type_counts: {aggregate['failure_type_counts']}")
    print(f"final_tilt_directions: {aggregate['final_tilt_directions']}")
    print(f"dominant_load_axes: {aggregate['dominant_load_axes']}")
    print(f"dominant_loaded_legs: {aggregate['dominant_loaded_legs']}")
    print(f"least_loaded_legs: {aggregate['least_loaded_legs']}")
    print("worst_case:")
    for key, value in aggregate["worst_case"].items():
        print(f"  {key}: {value:.6f}")
    settled = aggregate["window_means"]["settled_window"]
    print("settled_window:")
    for key in (
        "mean_abs_xz_vel",
        "base_displacement_from_reset",
        "base_displacement_from_active_ref",
        "mean_tilt_error",
        "total_contact_foot_slip_distance",
        "total_contact_switches",
        "min_foot_load",
        "max_foot_friction_usage",
        "max_foot_anchor_displacement",
        "total_foot_anchor_resets",
        "total_foot_anchor_deactivations",
        "action_pair_delta",
        "mean_abs_raw_action_delta",
        "mean_abs_executed_action_delta",
        "max_abs_executed_action_delta",
        "mean_filter_lag",
        "max_filter_lag",
    ):
        print(f"  {key}: {settled[key]:.6f}")


def main() -> None:
    args = parse_args()
    if not args.policy.exists():
        raise FileNotFoundError(f"Policy not found: {args.policy}")

    args.out.mkdir(parents=True, exist_ok=True)
    env = Go1Env(
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(args.friction_min, args.friction_max),
        reset_noise_level=args.reset_noise_level,
        reset_noise_components=args.reset_noise_components,
        spawn_x=args.spawn_x,
        spawn_z=args.spawn_z,
        ground_height_offset=args.ground_height_offset,
    )
    model = PPO.load(args.policy, device=SB3_DEVICE)

    timeline_file = None
    timeline_writer = None
    timeline_state = {"wrote_header": False}
    if args.log_every_step:
        timeline_file = (args.out / "timeline.csv").open("w", newline="", encoding="utf-8")
        timeline_writer = csv.DictWriter(timeline_file, fieldnames=[])

    try:
        episodes = [
            diagnose_episode(env, model, episode, args, timeline_writer, timeline_state)
            for episode in range(1, args.episodes + 1)
        ]
    finally:
        env.close()
        if timeline_file is not None:
            timeline_file.close()

    aggregate = aggregate_summaries(episodes, args)
    (args.out / "episodes.json").write_text(
        json.dumps(episodes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print_aggregate(aggregate)
    print(f"wrote: {args.out / 'summary.json'}")
    print(f"wrote: {args.out / 'episodes.json'}")
    if args.log_every_step:
        print(f"wrote: {args.out / 'timeline.csv'}")


if __name__ == "__main__":
    main()
