"""Run headless eval + diagnosis for one trained standing policy."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from project_config import DEFAULT_ACTION_FILTER_TAU


RESET_NOISE_LEVELS = ("clean", "rn1", "rn2", "rn3")
RESET_NOISE_COMPONENTS = ("combined", "joint_pos", "joint_vel", "roll_pitch", "yaw", "base_height", "base_position", "base_velocity")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="Path to a Stable-Baselines3 policy zip.")
    parser.add_argument("--name", required=True, help="Diagnostics output folder name.")
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.5)
    parser.add_argument("--friction-max", type=float, default=1.1)
    parser.add_argument("--ground-height-offset", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--reset-noise-level", choices=RESET_NOISE_LEVELS, default="clean")
    parser.add_argument("--reset-noise-components", choices=RESET_NOISE_COMPONENTS, default="combined")
    parser.add_argument("--early-window-steps", type=int, default=250)
    parser.add_argument("--settled-window-steps", type=int, default=250)
    parser.add_argument(
        "--action-filter-tau",
        type=float,
        default=DEFAULT_ACTION_FILTER_TAU,
        help="Environment action low-pass filter time constant in seconds.",
    )
    parser.add_argument(
        "--no-action-filter",
        action="store_true",
        help="Disable action filtering in spawned evaluation and diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=Path("diagnostics"),
        help="Root folder for regression outputs.",
    )
    args = parser.parse_args()
    if args.no_action_filter:
        args.action_filter_tau = None
    return args


def _run_command(command: list[str], output_path: Path) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    output_path.write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return output


def _literal_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def parse_evaluate_stdout(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    reward_terms: dict[str, float] = {}
    in_reward_terms = False

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line == "mean_reward_terms:":
            in_reward_terms = True
            continue
        if in_reward_terms and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            reward_terms[key] = float(value.strip())
            continue
        if ":" not in line or line.startswith("/"):
            continue
        in_reward_terms = False
        key, value = line.split(":", 1)
        value = value.strip()
        if key in {"episodes"}:
            metrics[key] = int(value)
        elif key in {
            "survival_rate",
            "mean_reward",
            "mean_length",
            "mean_abs_action",
            "max_abs_action",
            "min_base_relative_height",
            "min_upright_score",
            "friction_min_seen",
            "friction_max_seen",
        }:
            metrics[key] = float(value)
        elif key in {"min_axis_up", "termination_reasons"}:
            metrics[key] = _literal_value(value)

    metrics["mean_reward_terms"] = reward_terms
    return metrics


def _top_count(counts: dict[str, int] | None) -> dict[str, Any] | None:
    if not counts:
        return None
    key, value = max(counts.items(), key=lambda item: item[1])
    return {"key": key, "count": value}


def _settled_load_pair_metrics(settled_window: dict[str, Any]) -> dict[str, float | None]:
    shares = settled_window.get("foot_load_shares") or {}
    required_legs = ("FL", "FR", "RL", "RR")
    if not all(leg in shares for leg in required_legs):
        return {
            "settled_left_load_share": None,
            "settled_right_load_share": None,
            "settled_left_right_load_delta": None,
            "settled_front_load_share": None,
            "settled_rear_load_share": None,
            "settled_front_rear_load_delta": None,
        }

    left = float(shares["FL"] + shares["RL"])
    right = float(shares["FR"] + shares["RR"])
    front = float(shares["FL"] + shares["FR"])
    rear = float(shares["RL"] + shares["RR"])
    return {
        "settled_left_load_share": left,
        "settled_right_load_share": right,
        "settled_left_right_load_delta": abs(left - right),
        "settled_front_load_share": front,
        "settled_rear_load_share": rear,
        "settled_front_rear_load_delta": abs(front - rear),
    }


def compact_summary(
    args: argparse.Namespace,
    eval_metrics: dict[str, Any],
    diagnose_summary: dict[str, Any],
) -> dict[str, Any]:
    reward_terms = eval_metrics.get("mean_reward_terms", {})
    worst_case = diagnose_summary.get("worst_case", {})
    settled_window = diagnose_summary.get("window_means", {}).get("settled_window", {})
    load_pair_metrics = _settled_load_pair_metrics(settled_window)
    return {
        "name": args.name,
        "policy": str(args.policy),
        "terrain": args.terrain,
        "friction_range": [args.friction_min, args.friction_max],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "action_filter_tau": args.action_filter_tau,
        "reset_noise_level": args.reset_noise_level,
        "reset_noise_components": args.reset_noise_components,
        "ground_height_offset": args.ground_height_offset,
        "material_properties": diagnose_summary.get("material_properties"),
        "eval": {
            "survival_rate": eval_metrics.get("survival_rate"),
            "mean_length": eval_metrics.get("mean_length"),
            "mean_reward": eval_metrics.get("mean_reward"),
            "min_upright_score": eval_metrics.get("min_upright_score"),
            "min_base_relative_height": eval_metrics.get("min_base_relative_height"),
            "termination_reasons": eval_metrics.get("termination_reasons"),
            "friction_min_seen": eval_metrics.get("friction_min_seen"),
            "friction_max_seen": eval_metrics.get("friction_max_seen"),
            "effective_friction_min_seen": diagnose_summary.get("effective_friction_min_seen"),
            "effective_friction_max_seen": diagnose_summary.get("effective_friction_max_seen"),
            "foot_contact_error": reward_terms.get("foot_contact_error"),
            "min_foot_load": reward_terms.get("min_foot_load"),
            "mean_abs_xz_vel": reward_terms.get("mean_abs_xz_vel"),
            "max_abs_xz_vel": reward_terms.get("max_abs_xz_vel"),
            "tilt_error": reward_terms.get("tilt_error"),
            "trunk_stance_center_error": reward_terms.get("trunk_stance_center_error"),
            "trunk_stance_center_dx": reward_terms.get("trunk_stance_center_dx"),
            "trunk_stance_center_dz": reward_terms.get("trunk_stance_center_dz"),
            "leg_symmetry_error": reward_terms.get("leg_symmetry_error"),
            "settled_mean_abs_xz_vel": settled_window.get("mean_abs_xz_vel"),
            "settled_base_displacement_from_reset": settled_window.get("base_displacement_from_reset"),
            "settled_base_displacement_from_active_ref": settled_window.get("base_displacement_from_active_ref"),
            "settled_yaw_drift_from_reset": settled_window.get("yaw_drift_from_reset"),
            "settled_mean_trunk_x_up": settled_window.get("mean_trunk_x_up"),
            "settled_mean_trunk_y_up": settled_window.get("mean_trunk_y_up"),
            "settled_mean_base_relative_height": settled_window.get("mean_base_relative_height"),
            "settled_min_base_relative_height": settled_window.get("min_base_relative_height"),
            "settled_mean_tilt_error": settled_window.get("mean_tilt_error"),
            "settled_min_foot_load": settled_window.get("min_foot_load"),
            "settled_foot_tangential_force_mean": settled_window.get("foot_tangential_force_mean"),
            "settled_foot_tangential_force_max": settled_window.get("foot_tangential_force_max"),
            "settled_foot_friction_usage_mean": settled_window.get("foot_friction_usage_mean"),
            "settled_foot_friction_usage_max": settled_window.get("foot_friction_usage_max"),
            "settled_max_foot_friction_usage": settled_window.get("max_foot_friction_usage"),
            "settled_worst_friction_usage_foot": settled_window.get("worst_friction_usage_foot"),
            "settled_foot_friction_usage_above_0p5_frames": settled_window.get("foot_friction_usage_above_0p5_frames"),
            "settled_foot_friction_usage_above_0p7_frames": settled_window.get("foot_friction_usage_above_0p7_frames"),
            "settled_foot_friction_usage_above_0p9_frames": settled_window.get("foot_friction_usage_above_0p9_frames"),
            "settled_foot_friction_usage_above_1p0_frames": settled_window.get("foot_friction_usage_above_1p0_frames"),
            "settled_total_contact_foot_slip_distance": settled_window.get("total_contact_foot_slip_distance"),
            "settled_total_contact_switches": settled_window.get("total_contact_switches"),
            "settled_action_pair_delta": settled_window.get("action_pair_delta"),
            "settled_foot_load_shares": settled_window.get("foot_load_shares"),
            "settled_contact_duty": settled_window.get("contact_duty"),
            "settled_max_foot_anchor_displacement": settled_window.get("max_foot_anchor_displacement"),
            "settled_max_foot_anchor_error": settled_window.get("max_foot_anchor_error"),
            "settled_total_foot_anchor_resets": settled_window.get("total_foot_anchor_resets"),
            "settled_total_foot_anchor_deactivations": settled_window.get("total_foot_anchor_deactivations"),
            "settled_foot_anchor_active_duty": settled_window.get("foot_anchor_active_duty"),
            "settled_foot_anchor_reset_count": settled_window.get("foot_anchor_reset_count"),
            "settled_foot_anchor_deactivate_count": settled_window.get("foot_anchor_deactivate_count"),
            "settled_foot_load_below_20n_frames": settled_window.get("foot_load_below_20n_frames"),
            "settled_foot_load_below_15n_frames": settled_window.get("foot_load_below_15n_frames"),
            "settled_foot_load_below_8n_frames": settled_window.get("foot_load_below_8n_frames"),
            "settled_foot_load_below_5n_frames": settled_window.get("foot_load_below_5n_frames"),
            **load_pair_metrics,
        },
        "diagnosis": {
            "cause_counts": diagnose_summary.get("cause_counts"),
            "top_cause": _top_count(diagnose_summary.get("cause_counts")),
            "failure_type_counts": diagnose_summary.get("failure_type_counts"),
            "top_failure_type": _top_count(diagnose_summary.get("failure_type_counts")),
            "final_tilt_directions": diagnose_summary.get("final_tilt_directions"),
            "top_final_tilt_direction": _top_count(diagnose_summary.get("final_tilt_directions")),
            "dominant_load_axes": diagnose_summary.get("dominant_load_axes"),
            "top_dominant_load_axis": _top_count(diagnose_summary.get("dominant_load_axes")),
            "dominant_loaded_legs": diagnose_summary.get("dominant_loaded_legs"),
            "top_dominant_loaded_leg": _top_count(diagnose_summary.get("dominant_loaded_legs")),
            "least_loaded_legs": diagnose_summary.get("least_loaded_legs"),
            "top_least_loaded_leg": _top_count(diagnose_summary.get("least_loaded_legs")),
            "max_tilt_error": worst_case.get("max_tilt_error"),
            "max_load_imbalance": worst_case.get("max_load_imbalance"),
            "max_foot_dxz": worst_case.get("max_foot_dxz"),
            "max_trunk_stance_center_error": worst_case.get("max_trunk_stance_center_error"),
            "max_foot_anchor_displacement": worst_case.get("max_foot_anchor_displacement"),
            "max_foot_anchor_error": worst_case.get("max_foot_anchor_error"),
            "max_foot_anchor_resets": worst_case.get("max_foot_anchor_resets"),
            "max_foot_anchor_deactivations": worst_case.get("max_foot_anchor_deactivations"),
            "max_foot_friction_usage": worst_case.get("max_foot_friction_usage"),
            "max_nonfoot_load": worst_case.get("max_nonfoot_load"),
            "min_foot_load": worst_case.get("min_foot_load"),
            "min_upright_score": worst_case.get("min_upright_score"),
            "worst_episodes": diagnose_summary.get("worst_episodes"),
        },
    }


def run_condition(args: argparse.Namespace, out_dir: Path, name: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_command = [
        sys.executable,
        "evaluate_stand.py",
        str(args.policy),
        "--terrain",
        args.terrain,
        "--friction-min",
        str(args.friction_min),
        "--friction-max",
        str(args.friction_max),
        "--ground-height-offset",
        str(args.ground_height_offset),
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--reset-noise-level",
        args.reset_noise_level,
        "--reset-noise-components",
        args.reset_noise_components,
    ]
    if args.action_filter_tau is not None:
        eval_command.extend(["--action-filter-tau", str(args.action_filter_tau)])
    diagnose_command = [
        sys.executable,
        "diagnose_policy.py",
        str(args.policy),
        "--terrain",
        args.terrain,
        "--friction-min",
        str(args.friction_min),
        "--friction-max",
        str(args.friction_max),
        "--ground-height-offset",
        str(args.ground_height_offset),
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--early-window-steps",
        str(args.early_window_steps),
        "--settled-window-steps",
        str(args.settled_window_steps),
        "--reset-noise-level",
        args.reset_noise_level,
        "--reset-noise-components",
        args.reset_noise_components,
        "--out",
        str(out_dir),
    ]
    if args.action_filter_tau is not None:
        diagnose_command.extend(["--action-filter-tau", str(args.action_filter_tau)])

    print("Running evaluate_stand.py...")
    eval_stdout = _run_command(eval_command, out_dir / "evaluate_stdout.txt")
    eval_metrics = parse_evaluate_stdout(eval_stdout)

    print("Running diagnose_policy.py...")
    _run_command(diagnose_command, out_dir / "diagnose_stdout.txt")
    diagnose_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    summary_args = SimpleNamespace(**vars(args))
    summary_args.name = name
    summary = compact_summary(summary_args, eval_metrics, diagnose_summary)
    summary_path = out_dir / "regression_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    eval_block = summary["eval"]
    diagnosis_block = summary["diagnosis"]
    print(f"wrote: {summary_path}")
    print(
        "result: "
        f"survival={eval_block['survival_rate']:.3f} "
        f"length={eval_block['mean_length']:.1f} "
        f"reward={eval_block['mean_reward']:.3f} "
        f"upright={eval_block['min_upright_score']:.3f} "
        f"contact={eval_block['foot_contact_error']:.6f} "
        f"xz={eval_block['mean_abs_xz_vel']:.6f} "
        f"settled_lr_delta={eval_block['settled_left_right_load_delta']:.6f} "
        f"cause={diagnosis_block['top_cause']}"
    )
    return summary

def main() -> None:
    args = parse_args()
    if not args.policy.exists():
        raise FileNotFoundError(f"Policy not found: {args.policy}")

    out_dir = args.diagnostics_root / args.name
    run_condition(args, out_dir, args.name)


if __name__ == "__main__":
    main()
