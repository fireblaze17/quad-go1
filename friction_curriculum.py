"""Helper commands for the active flat-ground friction standing workflow.

By default this script prints the next command instead of running it. Pass
``--run`` when you intentionally want to start a long training/evaluation job.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from project_config import (
    CURRENT_BASELINE_MODEL,
    DEFAULT_ACTION_FILTER_TAU,
    FILTERED_FIXED_MODEL,
    FIXED_BASELINE_DIR,
    FIXED_BASELINE_MODEL,
    FRICTION_AB_DIR,
    FRICTION_A_DIR,
    ORIGINAL_STAND_MODEL,
)

BASE_SOURCE = ORIGINAL_STAND_MODEL
BASE_DIR = FIXED_BASELINE_DIR
BASE_MODEL = FIXED_BASELINE_MODEL


@dataclass(frozen=True)
class FrictionStage:
    name: str
    save_dir: Path
    load_model: Path
    friction_min: float
    friction_max: float
    timesteps: int = 300_000

    @property
    def model_path(self) -> Path:
        return self.save_dir / "final_model.zip"


STAGES = {
    "friction_a": FrictionStage(
        name="friction_a",
        save_dir=FRICTION_A_DIR,
        load_model=BASE_MODEL,
        friction_min=0.7,
        friction_max=0.9,
    ),
}


def _python() -> str:
    return sys.executable


def _path(path: Path) -> str:
    return str(path)


def train_command(stage: FrictionStage) -> list[str]:
    return [
        _python(),
        "train_stand.py",
        "--terrain",
        "flat",
        "--friction-min",
        str(stage.friction_min),
        "--friction-max",
        str(stage.friction_max),
        "--load",
        _path(stage.load_model),
        "--save-dir",
        _path(stage.save_dir),
        "--timesteps",
        str(stage.timesteps),
    ]


def fixed_clean_train_command() -> list[str]:
    return friction_bridge_commands()[0]


def friction_bridge_commands() -> list[list[str]]:
    commands = []
    for mu in (0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2):
        commands.append([
            _python(),
            "diagnose_policy.py",
            _path(CURRENT_BASELINE_MODEL),
            "--terrain",
            "flat",
            "--friction-min",
            str(mu),
            "--friction-max",
            str(mu),
            "--episodes",
            "30",
            "--max-steps",
            "5000",
            "--action-filter-tau",
            str(DEFAULT_ACTION_FILTER_TAU),
            "--out",
            f"diagnostics/friction_slice_current10k_mu_{mu:.1f}".replace(".", "p"),
        ])
    return commands


def friction_bridge_stress_commands() -> list[list[str]]:
    commands = []
    for mu in (0.45, 1.3):
        commands.append([
            _python(),
            "diagnose_policy.py",
            _path(CURRENT_BASELINE_MODEL),
            "--terrain",
            "flat",
            "--friction-min",
            str(mu),
            "--friction-max",
            str(mu),
            "--episodes",
            "30",
            "--max-steps",
            "5000",
            "--action-filter-tau",
            str(DEFAULT_ACTION_FILTER_TAU),
            "--out",
            f"diagnostics/friction_stress_current10k_mu_{mu:.2f}".replace(".", "p"),
        ])
    return commands


def friction_randomization_train_command() -> list[str]:
    return [
        _python(),
        "train_stand.py",
        "--terrain",
        "flat",
        "--friction-min",
        "0.5",
        "--friction-max",
        "1.2",
        "--load",
        _path(FILTERED_FIXED_MODEL),
        "--save-dir",
        "runs/stand_friction_random_050_120_tau005_from_filtered2k_lowlr",
        "--timesteps",
        "20000",
        "--seed",
        "1",
        "--checkpoint-freq",
        "2000",
        "--learning-rate",
        "0.000025",
        "--clip-range",
        "0.05",
        "--target-kl",
        "0.005",
        "--action-filter-tau",
        str(DEFAULT_ACTION_FILTER_TAU),
    ]


def eval_command(model: Path, friction_min: float, friction_max: float, episodes: int) -> list[str]:
    return [
        _python(),
        "evaluate_stand.py",
        _path(model),
        "--terrain",
        "flat",
        "--friction-min",
        str(friction_min),
        "--friction-max",
        str(friction_max),
        "--episodes",
        str(episodes),
    ]


def view_command(stage: FrictionStage) -> list[str]:
    return [
        _python(),
        "view_stand_policy.py",
        _path(stage.model_path),
        "--terrain",
        "flat",
        "--friction-min",
        str(stage.friction_min),
        "--friction-max",
        str(stage.friction_max),
    ]


def prepare_base() -> None:
    if not BASE_SOURCE.exists():
        raise FileNotFoundError(f"Missing accepted base model: {BASE_SOURCE}")

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_SOURCE, BASE_MODEL)

    copied_metadata = False
    for name in ("args.json", "env_constants.json"):
        source = BASE_SOURCE.parent / name
        if source.exists():
            shutil.copy2(source, BASE_DIR / name)
            copied_metadata = True

    if not copied_metadata:
        notes = BASE_DIR / "baseline_notes.md"
        notes.write_text(
            "# Standing Base V2\n\n"
            "This directory preserves the accepted fixed-friction standing v2 "
            "checkpoint before randomized-friction curriculum training.\n\n"
            f"Source model: `{BASE_SOURCE.as_posix()}`\n"
            "Terrain: flat\n"
            "Friction: 0.8 fixed\n",
            encoding="utf-8",
        )

    print(f"prepared base model: {BASE_MODEL}")


def print_command(command: list[str]) -> None:
    print(" ".join(command))


def run_or_print(command: list[str], run: bool) -> None:
    print_command(command)
    if run:
        check_pychrono_parser_access()
        subprocess.run(command, check=True)


def check_pychrono_parser_access() -> None:
    try:
        importlib.import_module("pychrono.parsers")
    except ImportError as exc:
        raise SystemExit(
            "PyChrono parser import failed before the curriculum command could run.\n"
            "Use the WSL chrono-go1 environment documented in README.md and "
            "docs/reproducibility.md. If this happens in native Windows, Smart "
            "App Control may be blocking Chrono extension binaries; WSL is the "
            "project's supported path.\n\n"
            f"Original error: {exc}"
        ) from exc


def print_status() -> None:
    for path in (
        CURRENT_BASELINE_MODEL,
        BASE_MODEL,
        FRICTION_A_DIR / "final_model.zip",
        FRICTION_AB_DIR / "final_model.zip",
    ):
        state = "exists" if path.exists() else "missing"
        print(f"{state:<7} {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare-base",
            "train",
            "eval",
            "view",
            "status",
            "all-commands",
            "fixed-clean",
            "bridge-check",
            "bridge-stress",
            "friction-randomization",
        ],
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=list(STAGES),
        help="Curriculum stage for train/eval/view.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--run", action="store_true", help="Run the command instead of only printing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.action == "prepare-base":
        prepare_base()
        return

    if args.action == "status":
        print_status()
        return

    if args.action == "fixed-clean":
        print("# fixed-clean is retained as a compatibility alias for bridge-check.")
        for command in friction_bridge_commands():
            run_or_print(command, args.run)
        return

    if args.action == "bridge-check":
        for command in friction_bridge_commands():
            run_or_print(command, args.run)
        return

    if args.action == "bridge-stress":
        for command in friction_bridge_stress_commands():
            run_or_print(command, args.run)
        return

    if args.action == "friction-randomization":
        print("# Fallback only: current 0.5-1.2 baseline is already accepted.")
        print("# Run this only if a future material/reward change breaks fixed slices.")
        run_or_print(friction_randomization_train_command(), args.run)
        return

    if args.action == "all-commands":
        print("# Current accepted-baseline fixed-slice checks")
        for command in friction_bridge_commands():
            print_command(command)
        print("\n# Optional outside-range stress checks")
        for command in friction_bridge_stress_commands():
            print_command(command)
        print("\n# Fallback low-LR friction randomization if future fixed slices regress")
        print_command(friction_randomization_train_command())
        print("\n# Archived pre-filter friction-A helper commands")
        print_command([_python(), "friction_curriculum.py", "prepare-base"])
        for stage in STAGES.values():
            print(f"# Archived train {stage.name}")
            print_command(train_command(stage))
            print(f"# Archived evaluate {stage.name}")
            print_command(eval_command(stage.model_path, stage.friction_min, stage.friction_max, 10))
            print(f"# Archived view {stage.name}")
            print_command(view_command(stage))
        return

    if args.stage is None:
        raise SystemExit(f"{args.action} requires a stage: {', '.join(STAGES)}")

    stage = STAGES[args.stage]
    if args.action == "train":
        run_or_print(train_command(stage), args.run)
    elif args.action == "eval":
        run_or_print(
            eval_command(stage.model_path, stage.friction_min, stage.friction_max, args.episodes),
            args.run,
        )
    elif args.action == "view":
        run_or_print(view_command(stage), args.run)


if __name__ == "__main__":
    main()
