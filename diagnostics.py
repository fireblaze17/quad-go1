"""Shared viewer diagnostics for Chrono Go1 foot and contact behavior."""

from __future__ import annotations

from typing import Any


FOOT_BODY_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
LEG_PREFIXES = ("FR", "FL", "RR", "RL")
CONTACT_BODY_GROUPS = ("foot", "calf", "thigh", "hip")


def bodies_by_name(env) -> dict[str, Any]:
    return {body.GetName(): body for body in env._system.GetBodies()}


def foot_bodies(env) -> list[Any]:
    lookup = bodies_by_name(env)
    return [lookup[name] for name in FOOT_BODY_NAMES]


def contact_body_groups(env) -> dict[str, list[Any]]:
    lookup = bodies_by_name(env)
    return {
        group: [lookup[f"{leg}_{group}"] for leg in LEG_PREFIXES]
        for group in CONTACT_BODY_GROUPS
    }


def foot_xz_positions(feet: list[Any]) -> list[tuple[float, float]]:
    return [(float(body.GetPos().x), float(body.GetPos().z)) for body in feet]


def body_loads_y(bodies: list[Any]) -> list[float]:
    return [abs(float(body.GetContactForce().y)) for body in bodies]


def foot_debug_stats(
    feet: list[Any],
    reset_xz: list[tuple[float, float]],
) -> dict[str, Any]:
    displacements = []
    speeds = []
    heights = []
    contact_loads = []

    for body, (reset_x, reset_z) in zip(feet, reset_xz):
        pos = body.GetPos()
        vel = body.GetPosDt()
        dx = float(pos.x) - reset_x
        dz = float(pos.z) - reset_z
        displacements.append((dx * dx + dz * dz) ** 0.5)
        speeds.append((float(vel.x) ** 2 + float(vel.z) ** 2) ** 0.5)
        heights.append(float(pos.y))
        contact_loads.append(abs(float(body.GetContactForce().y)))

    total_load = sum(contact_loads)
    if total_load > 1e-9:
        load_shares = [load / total_load for load in contact_loads]
        load_imbalance = max(load_shares) - min(load_shares)
    else:
        load_shares = [0.0 for _ in contact_loads]
        load_imbalance = 0.0

    return {
        "foot_dxz_mean": sum(displacements) / len(displacements),
        "foot_dxz_max": max(displacements),
        "foot_vxz_mean": sum(speeds) / len(speeds),
        "foot_vxz_max": max(speeds),
        "foot_heights": heights,
        "foot_contact_loads": contact_loads,
        "foot_load_shares": load_shares,
        "foot_load_imbalance": load_imbalance,
        "foot_displacements": displacements,
        "foot_speeds": speeds,
    }


def new_interval_stats() -> dict[str, list[float]]:
    return {
        "foot_load_min": [float("inf")] * len(FOOT_BODY_NAMES),
        "foot_load_max": [0.0] * len(FOOT_BODY_NAMES),
        "foot_y_min": [float("inf")] * len(FOOT_BODY_NAMES),
        "foot_y_max": [float("-inf")] * len(FOOT_BODY_NAMES),
        "nonfoot_load_max": [0.0] * len(FOOT_BODY_NAMES),
    }


def update_interval_stats(
    interval_stats: dict[str, list[float]],
    foot_stats: dict[str, Any],
    contact_stats: dict[str, Any],
) -> None:
    for i, (height, load) in enumerate(
        zip(foot_stats["foot_heights"], foot_stats["foot_contact_loads"])
    ):
        interval_stats["foot_y_min"][i] = min(interval_stats["foot_y_min"][i], height)
        interval_stats["foot_y_max"][i] = max(interval_stats["foot_y_max"][i], height)
        interval_stats["foot_load_min"][i] = min(
            interval_stats["foot_load_min"][i],
            load,
        )
        interval_stats["foot_load_max"][i] = max(
            interval_stats["foot_load_max"][i],
            load,
        )
    for i, load in enumerate(contact_stats["nonfoot_loads"]):
        interval_stats["nonfoot_load_max"][i] = max(
            interval_stats["nonfoot_load_max"][i],
            load,
        )


def contact_debug_stats(groups: dict[str, list[Any]]) -> dict[str, Any]:
    group_loads = {group: body_loads_y(bodies) for group, bodies in groups.items()}
    nonfoot_loads = [
        sum(group_loads[group][i] for group in ("calf", "thigh", "hip"))
        for i in range(len(LEG_PREFIXES))
    ]
    return {
        "group_loads": group_loads,
        "nonfoot_loads": nonfoot_loads,
    }


def format_foot_values(label: str, values: list[float], digits: int = 3) -> str:
    entries = [
        f"{name.split('_')[0]}:{value:+.{digits}f}"
        for name, value in zip(FOOT_BODY_NAMES, values)
    ]
    return f"{label}=(" + ",".join(entries) + ")"
