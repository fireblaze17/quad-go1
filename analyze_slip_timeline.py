"""Summarize settled-window foot slip from diagnose_policy timeline CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


_LEGS = ("FL", "FR", "RL", "RR")
_TIME_STEP = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path, help="Path to timeline.csv from diagnose_policy.py.")
    parser.add_argument(
        "--settled-window-steps",
        type=int,
        default=250,
        help="Number of final steps per episode to analyze.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=10.0,
        help="Foot load in N counted as contact for slip/contact metrics.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for machine-readable summary JSON.",
    )
    return parser.parse_args()


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _json_map(row: dict[str, str], key: str) -> dict[str, float]:
    raw = row.get(key) or "{}"
    value = json.loads(raw)
    return {str(item_key): float(item_value) for item_key, item_value in value.items()}


def _json_position_map(row: dict[str, str], key: str) -> dict[str, tuple[float, float]]:
    raw = row.get(key) or "{}"
    value = json.loads(raw)
    return {
        str(item_key): (float(item_value[0]), float(item_value[1]))
        for item_key, item_value in value.items()
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_by_episode(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    episodes: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        episode = int(float(row.get("episode", 0)))
        episodes.setdefault(episode, []).append(row)
    return episodes


def _dot(ax: float, az: float, bx: float, bz: float) -> float:
    return ax * bx + az * bz


def _norm(x: float, z: float) -> float:
    return math.sqrt(x * x + z * z)


def _cosine_similarity(ax: float, az: float, bx: float, bz: float) -> float | None:
    a_norm = _norm(ax, az)
    b_norm = _norm(bx, bz)
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return None
    return _dot(ax, az, bx, bz) / (a_norm * b_norm)


def _contact_switch_count(contacts: list[bool]) -> int:
    if not contacts:
        return 0
    return sum(int(current != previous) for previous, current in zip(contacts, contacts[1:]))


def _summarize_episode(
    rows: list[dict[str, str]],
    settled_window_steps: int,
    contact_threshold: float,
) -> dict[str, Any]:
    settled = rows[-settled_window_steps:] if settled_window_steps > 0 else rows
    if not settled:
        raise ValueError("timeline has no rows to analyze")

    first = settled[0]
    last = settled[-1]
    base_dx = _as_float(last, "base_x") - _as_float(first, "base_x")
    base_dz = _as_float(last, "base_z") - _as_float(first, "base_z")
    base_displacement = _norm(base_dx, base_dz)
    mean_abs_xz_vel = mean(
        (abs(_as_float(row, "lin_vel_x")) + abs(_as_float(row, "lin_vel_z"))) / 2.0
        for row in settled
    )

    per_foot: dict[str, dict[str, float | int | None]] = {}
    total_slip = 0.0
    loaded_speed_values: list[float] = []
    displacement_values: list[float] = []
    low_load_slip = 0.0
    high_load_slip = 0.0

    first_displacements = _json_map(first, "foot_displacements")
    last_displacements = _json_map(last, "foot_displacements")
    first_positions = _json_position_map(first, "foot_positions") if first.get("foot_positions") else {}
    last_positions = _json_position_map(last, "foot_positions") if last.get("foot_positions") else {}

    for leg in _LEGS:
        loads = [_json_map(row, "foot_loads").get(leg, 0.0) for row in settled]
        speeds = [_json_map(row, "foot_speeds").get(leg, 0.0) for row in settled]
        contacts = [load > contact_threshold for load in loads]
        loaded_speeds = [speed for speed, contact in zip(speeds, contacts) if contact]
        slip_distance = sum(speed * _TIME_STEP for speed, contact in zip(speeds, contacts) if contact)
        low_load_slip += sum(
            speed * _TIME_STEP
            for speed, load in zip(speeds, loads)
            if contact_threshold < load < 20.0
        )
        high_load_slip += sum(
            speed * _TIME_STEP
            for speed, load in zip(speeds, loads)
            if load >= 20.0
        )
        if leg in first_positions and leg in last_positions:
            foot_dx = last_positions[leg][0] - first_positions[leg][0]
            foot_dz = last_positions[leg][1] - first_positions[leg][1]
            world_displacement = _norm(foot_dx, foot_dz)
        else:
            foot_dx = 0.0
            foot_dz = 0.0
            world_displacement = last_displacements.get(leg, 0.0) - first_displacements.get(leg, 0.0)
        mean_loaded_speed = mean(loaded_speeds) if loaded_speeds else 0.0
        motion_vs_base = _cosine_similarity(foot_dx, foot_dz, base_dx, base_dz)

        total_slip += slip_distance
        loaded_speed_values.extend(loaded_speeds)
        displacement_values.append(world_displacement)
        per_foot[leg] = {
            "settled_slip_distance": slip_distance,
            "settled_world_dx": foot_dx,
            "settled_world_dz": foot_dz,
            "settled_world_displacement": world_displacement,
            "mean_loaded_speed": mean_loaded_speed,
            "max_loaded_speed": max(loaded_speeds) if loaded_speeds else 0.0,
            "mean_load": mean(loads),
            "min_load": min(loads),
            "max_load": max(loads),
            "contact_duty": sum(contacts) / len(contacts),
            "contact_switches": _contact_switch_count(contacts),
            "motion_vs_base_cosine": motion_vs_base,
        }

    dominant_slip_leg = max(per_foot, key=lambda leg: float(per_foot[leg]["settled_slip_distance"]))
    max_leg_slip = float(per_foot[dominant_slip_leg]["settled_slip_distance"])
    total_switches = sum(int(per_foot[leg]["contact_switches"]) for leg in _LEGS)
    all_feet_contact = all(float(per_foot[leg]["contact_duty"]) >= 0.99 for leg in _LEGS)
    mean_loaded_speed = mean(loaded_speed_values) if loaded_speed_values else 0.0
    max_displacement = max(displacement_values) if displacement_values else 0.0

    if total_slip <= 0.05:
        classification = "nominal"
    elif max_leg_slip >= 0.45 * total_slip:
        classification = "one_foot_creep"
    elif all_feet_contact and total_switches == 0 and max_displacement <= 0.01:
        classification = "contact_jitter"
    elif all_feet_contact and total_switches == 0 and base_displacement >= 0.005:
        classification = "all_feet_creep"
    elif low_load_slip > high_load_slip:
        classification = "unloaded_slip"
    else:
        classification = "mixed"

    return {
        "steps": len(settled),
        "body": {
            "base_dx": base_dx,
            "base_dz": base_dz,
            "base_displacement": base_displacement,
            "mean_abs_xz_vel": mean_abs_xz_vel,
        },
        "per_foot": per_foot,
        "totals": {
            "settled_total_slip_distance": total_slip,
            "dominant_slip_leg": dominant_slip_leg,
            "dominant_slip_fraction": max_leg_slip / total_slip if total_slip > 0 else 0.0,
            "total_contact_switches": total_switches,
            "mean_loaded_speed": mean_loaded_speed,
            "max_foot_displacement_delta": max_displacement,
            "low_load_slip": low_load_slip,
            "high_load_slip": high_load_slip,
        },
        "classification": classification,
    }


def _aggregate(episode_summaries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    items = list(episode_summaries.values())
    if not items:
        return {}

    classifications: dict[str, int] = {}
    for item in items:
        label = str(item["classification"])
        classifications[label] = classifications.get(label, 0) + 1

    return {
        "episodes": len(items),
        "classification_counts": classifications,
        "mean_total_slip": mean(item["totals"]["settled_total_slip_distance"] for item in items),
        "mean_base_displacement": mean(item["body"]["base_displacement"] for item in items),
        "mean_total_contact_switches": mean(item["totals"]["total_contact_switches"] for item in items),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    print(f"episodes: {aggregate['episodes']}")
    print(f"classification_counts: {aggregate['classification_counts']}")
    print(f"mean_total_slip: {aggregate['mean_total_slip']:.6f}")
    print(f"mean_base_displacement: {aggregate['mean_base_displacement']:.6f}")
    print(f"mean_total_contact_switches: {aggregate['mean_total_contact_switches']:.3f}")

    for episode, item in summary["episodes"].items():
        print(f"\nepisode {episode}: {item['classification']}")
        print(
            "body: "
            f"dx={item['body']['base_dx']:.6f} "
            f"dz={item['body']['base_dz']:.6f} "
            f"disp={item['body']['base_displacement']:.6f} "
            f"mean_abs_xz_vel={item['body']['mean_abs_xz_vel']:.6f}"
        )
        print(
            "totals: "
            f"slip={item['totals']['settled_total_slip_distance']:.6f} "
            f"dominant={item['totals']['dominant_slip_leg']} "
            f"fraction={item['totals']['dominant_slip_fraction']:.3f} "
            f"switches={item['totals']['total_contact_switches']} "
            f"low_load_slip={item['totals']['low_load_slip']:.6f} "
            f"high_load_slip={item['totals']['high_load_slip']:.6f}"
        )
        for leg in _LEGS:
            foot = item["per_foot"][leg]
            print(
                f"  {leg}: "
                f"slip={foot['settled_slip_distance']:.6f} "
                f"dx={foot['settled_world_dx']:.6f} "
                f"dz={foot['settled_world_dz']:.6f} "
                f"disp={foot['settled_world_displacement']:.6f} "
                f"mean_speed={foot['mean_loaded_speed']:.6f} "
                f"max_speed={foot['max_loaded_speed']:.6f} "
                f"mean_load={foot['mean_load']:.3f} "
                f"min_load={foot['min_load']:.3f} "
                f"duty={foot['contact_duty']:.3f} "
                f"switches={foot['contact_switches']} "
                f"base_cos={foot['motion_vs_base_cosine']}"
            )


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.timeline)
    by_episode = _group_by_episode(rows)
    episode_summaries = {
        episode: _summarize_episode(rows, args.settled_window_steps, args.contact_threshold)
        for episode, rows in sorted(by_episode.items())
    }
    summary = {
        "timeline": str(args.timeline),
        "settled_window_steps": args.settled_window_steps,
        "contact_threshold": args.contact_threshold,
        "aggregate": _aggregate(episode_summaries),
        "episodes": episode_summaries,
    }
    _print_summary(summary)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
