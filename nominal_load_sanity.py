"""Measure natural Go1 standing load shares under zero-action home-pose hold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from diagnostics import FOOT_BODY_NAMES, foot_bodies, foot_debug_stats, foot_xz_positions
from go1_env import Go1Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.8)
    parser.add_argument("--friction-max", type=float, default=0.8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--settled-window-steps", type=int, default=250)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/nominal_load_sanity"))
    return parser.parse_args()


def _foot_map(values: list[float]) -> dict[str, float]:
    return {
        name.split("_")[0]: float(value)
        for name, value in zip(FOOT_BODY_NAMES, values)
    }


def _mean_foot_map(items: list[dict[str, float]]) -> dict[str, float]:
    legs = [name.split("_")[0] for name in FOOT_BODY_NAMES]
    return {
        leg: float(np.mean([item[leg] for item in items])) if items else 0.0
        for leg in legs
    }


def run_episode(env: Go1Env, episode: int, args: argparse.Namespace) -> dict[str, Any]:
    obs, info = env.reset()
    feet = foot_bodies(env)
    reset_xz = foot_xz_positions(feet)
    shares = []
    loads = []
    action = np.zeros(12, dtype=np.float32)
    done = False
    steps = 0

    while not done:
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        stats = foot_debug_stats(feet, reset_xz)
        if steps > max(0, args.max_steps - args.settled_window_steps):
            shares.append(_foot_map(stats["foot_load_shares"]))
            loads.append(_foot_map(stats["foot_contact_loads"]))
        done = bool(terminated or truncated)

    return {
        "episode": episode,
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "friction": info.get("ground_friction"),
        "settled_foot_load_shares": _mean_foot_map(shares),
        "settled_foot_loads": _mean_foot_map(loads),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = Go1Env(
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(args.friction_min, args.friction_max),
    )
    try:
        episodes = [
            run_episode(env, episode, args)
            for episode in range(1, args.episodes + 1)
        ]
    finally:
        env.close()

    summary = {
        "terrain": args.terrain,
        "friction_range": [args.friction_min, args.friction_max],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "settled_window_steps": args.settled_window_steps,
        "mean_settled_foot_load_shares": _mean_foot_map(
            [item["settled_foot_load_shares"] for item in episodes]
        ),
        "mean_settled_foot_loads": _mean_foot_map(
            [item["settled_foot_loads"] for item in episodes]
        ),
    }
    (args.out / "episodes.json").write_text(
        json.dumps(episodes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {args.out / 'summary.json'}")
    print(f"mean_settled_foot_load_shares: {summary['mean_settled_foot_load_shares']}")


if __name__ == "__main__":
    main()
