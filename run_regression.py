"""Run headless eval + diagnosis for one trained standing policy."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="Path to a Stable-Baselines3 policy zip.")
    parser.add_argument("--name", required=True, help="Diagnostics output folder name.")
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.5)
    parser.add_argument("--friction-max", type=float, default=1.1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=Path("diagnostics"),
        help="Root folder for regression outputs.",
    )
    return parser.parse_args()


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
            "min_trunk_y",
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


def compact_summary(
    args: argparse.Namespace,
    eval_metrics: dict[str, Any],
    diagnose_summary: dict[str, Any],
) -> dict[str, Any]:
    reward_terms = eval_metrics.get("mean_reward_terms", {})
    worst_case = diagnose_summary.get("worst_case", {})
    return {
        "name": args.name,
        "policy": str(args.policy),
        "terrain": args.terrain,
        "friction_range": [args.friction_min, args.friction_max],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "eval": {
            "survival_rate": eval_metrics.get("survival_rate"),
            "mean_length": eval_metrics.get("mean_length"),
            "mean_reward": eval_metrics.get("mean_reward"),
            "min_upright_score": eval_metrics.get("min_upright_score"),
            "min_trunk_y": eval_metrics.get("min_trunk_y"),
            "termination_reasons": eval_metrics.get("termination_reasons"),
            "friction_min_seen": eval_metrics.get("friction_min_seen"),
            "friction_max_seen": eval_metrics.get("friction_max_seen"),
            "foot_contact_error": reward_terms.get("foot_contact_error"),
            "min_foot_load": reward_terms.get("min_foot_load"),
            "mean_abs_xz_vel": reward_terms.get("mean_abs_xz_vel"),
            "max_abs_xz_vel": reward_terms.get("max_abs_xz_vel"),
            "tilt_error": reward_terms.get("tilt_error"),
            "leg_symmetry_error": reward_terms.get("leg_symmetry_error"),
        },
        "diagnosis": {
            "cause_counts": diagnose_summary.get("cause_counts"),
            "top_cause": _top_count(diagnose_summary.get("cause_counts")),
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
            "max_nonfoot_load": worst_case.get("max_nonfoot_load"),
            "min_foot_load": worst_case.get("min_foot_load"),
            "min_upright_score": worst_case.get("min_upright_score"),
        },
    }


def main() -> None:
    args = parse_args()
    if not args.policy.exists():
        raise FileNotFoundError(f"Policy not found: {args.policy}")

    out_dir = args.diagnostics_root / args.name
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
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
    ]
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
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--out",
        str(out_dir),
    ]

    print("Running evaluate_stand.py...")
    eval_stdout = _run_command(eval_command, out_dir / "evaluate_stdout.txt")
    eval_metrics = parse_evaluate_stdout(eval_stdout)

    print("Running diagnose_policy.py...")
    _run_command(diagnose_command, out_dir / "diagnose_stdout.txt")
    diagnose_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    summary = compact_summary(args, eval_metrics, diagnose_summary)
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
        f"cause={diagnosis_block['top_cause']}"
    )


if __name__ == "__main__":
    main()
