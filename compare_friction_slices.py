"""Compare two standing policies across fixed friction slices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from project_config import DEFAULT_ACTION_FILTER_TAU

DEFAULT_MUS = (0.50, 0.60, 0.70, 0.80, 0.95, 1.10)
RESET_NOISE_LEVELS = ("clean", "rn1", "rn2", "rn3")
RESET_NOISE_COMPONENTS = ("combined", "joint_pos", "joint_vel", "roll_pitch", "yaw", "base_height", "base_position", "base_velocity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--ground-height-offset", type=float, default=0.0)
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
        help="Disable action filtering in spawned diagnostics.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        nargs="*",
        default=list(DEFAULT_MUS),
        help="Fixed friction values to test.",
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=Path("diagnostics"),
    )
    args = parser.parse_args()
    if args.no_action_filter:
        args.action_filter_tau = None
    return args


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(label: str, mu: float, summary: dict[str, Any], split: str) -> dict[str, Any]:
    eval_block = summary["eval"]
    diagnosis = summary["diagnosis"]
    return {
        "policy_label": label,
        "mu": mu,
        "split": split,
        "reset_noise_level": summary.get("reset_noise_level"),
        "reset_noise_components": summary.get("reset_noise_components"),
        "ground_height_offset": summary.get("ground_height_offset"),
        "survival_rate": eval_block.get("survival_rate"),
        "mean_reward": eval_block.get("mean_reward"),
        "mean_abs_xz_vel": eval_block.get("mean_abs_xz_vel"),
        "settled_mean_abs_xz_vel": eval_block.get("settled_mean_abs_xz_vel"),
        "settled_base_displacement_from_reset": eval_block.get("settled_base_displacement_from_reset"),
        "settled_mean_tilt_error": eval_block.get("settled_mean_tilt_error"),
        "settled_total_contact_foot_slip_distance": eval_block.get("settled_total_contact_foot_slip_distance"),
        "settled_total_contact_switches": eval_block.get("settled_total_contact_switches"),
        "settled_min_foot_load": eval_block.get("settled_min_foot_load"),
        "settled_foot_load_shares": eval_block.get("settled_foot_load_shares"),
        "top_failure_type": diagnosis.get("top_failure_type"),
        "top_dominant_loaded_leg": diagnosis.get("top_dominant_loaded_leg"),
        "top_least_loaded_leg": diagnosis.get("top_least_loaded_leg"),
    }


def main() -> None:
    args = parse_args()
    for path in (args.baseline, args.candidate):
        if not path.exists():
            raise FileNotFoundError(path)

    base_out = args.diagnostics_root / args.name
    rows = []
    for label, policy in (("baseline", args.baseline), ("candidate", args.candidate)):
        for mu in args.mu:
            run_name = f"{args.name}_{label}_mu_{mu:.2f}".replace(".", "p")
            command = [
                sys.executable,
                "run_regression.py",
                str(policy),
                "--name",
                run_name,
                "--terrain",
                args.terrain,
                "--friction-min",
                str(mu),
                "--friction-max",
                str(mu),
                "--episodes",
                str(args.episodes),
                "--max-steps",
                str(args.max_steps),
                "--ground-height-offset",
                str(args.ground_height_offset),
                "--reset-noise-level",
                args.reset_noise_level,
                "--reset-noise-components",
                args.reset_noise_components,
                "--early-window-steps",
                str(args.early_window_steps),
                "--settled-window-steps",
                str(args.settled_window_steps),
                "--diagnostics-root",
                str(args.diagnostics_root),
            ]
            if args.action_filter_tau is not None:
                command.extend(["--action-filter-tau", str(args.action_filter_tau)])
            _run(command)
            summary_dir = args.diagnostics_root / run_name
            rows.append(_row(label, mu, _load_summary(summary_dir / "regression_summary.json"), "friction_only"))

    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "friction_slice_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {base_out / 'friction_slice_rows.json'}")


if __name__ == "__main__":
    main()
