"""Evaluate command-space failures for two SCM locomotion checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch

from go1_scm_env import Go1SCMEnv, scm_env_metadata
from ppo_compat import EnvClippedActionPPO, load_ppo_same_shape_action_space
from project_config import SB3_DEVICE


DEFAULT_BASELINE_POLICY = Path(
    "runs/default_scm_finetune_lr3e5_clip005_from516m_v1_100m_continue/"
    "checkpoints/stand_policy_570990864_steps.zip"
)
DEFAULT_LATEST_POLICY = Path(
    "runs/default_scm_finetune_lr3e5_clip005_kl003_from571m_v1_100m_continue/"
    "checkpoints/stand_policy_628989936_steps.zip"
)
DEFAULT_OUT = Path("diagnostics/command_failure_map_571m_vs_latest")

VX_BINS = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]
VZ_BINS = [(-0.6, -0.36), (-0.36, -0.12), (-0.12, 0.12), (0.12, 0.36), (0.36, 0.6)]
YAW_BINS = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-policy", type=Path, default=DEFAULT_BASELINE_POLICY)
    parser.add_argument("--latest-policy", type=Path, default=DEFAULT_LATEST_POLICY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--episodes-per-bin", type=int, default=5)
    parser.add_argument("--standing-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--command-seed", type=int, default=20260812)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--limit-bins", type=int, default=0, help="Smoke-test only: evaluate the first N moving bins.")
    parser.add_argument("--device", default=SB3_DEVICE)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--worker-start-method", choices=("fork", "forkserver", "spawn"), default="forkserver")
    return parser.parse_args()


def _checkpoint_id(path: Path) -> str:
    match = re.search(r"stand_policy_(\d+)_steps\.zip", path.name)
    if match:
        return match.group(1)
    return path.stem


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
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    extra = sorted({key for row in rows for key in row} - set(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames + extra})


def _sample_command(rng: np.random.Generator, bins: tuple[int, int, int]) -> tuple[float, float, float]:
    vx_i, vz_i, yaw_i = bins
    return (
        float(rng.uniform(*VX_BINS[vx_i])),
        float(rng.uniform(*VZ_BINS[vz_i])),
        float(rng.uniform(*YAW_BINS[yaw_i])),
    )


def _set_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    torch.set_num_threads(1)


def build_episode_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.command_seed)
    plan: list[dict[str, Any]] = []
    moving_bins = [(vx, vz, yaw) for vx in range(5) for vz in range(5) for yaw in range(5)]
    if args.limit_bins > 0:
        moving_bins = moving_bins[: args.limit_bins]

    episode_index = 0
    for bins in moving_bins:
        for repeat in range(max(0, args.episodes_per_bin)):
            vx, vz, yaw = _sample_command(rng, bins)
            plan.append(
                {
                    "episode_index": episode_index,
                    "seed": int(args.seed_base + episode_index),
                    "vx": vx,
                    "vz": vz,
                    "yaw_rate": yaw,
                    "vx_bin": bins[0],
                    "vz_bin": bins[1],
                    "yaw_bin": bins[2],
                    "bin_kind": "moving",
                    "repeat": repeat,
                }
            )
            episode_index += 1

    for repeat in range(max(0, args.standing_episodes)):
        plan.append(
            {
                "episode_index": episode_index,
                "seed": int(args.seed_base + episode_index),
                "vx": 0.0,
                "vz": 0.0,
                "yaw_rate": 0.0,
                "vx_bin": "stand",
                "vz_bin": "stand",
                "yaw_bin": "stand",
                "bin_kind": "standing",
                "repeat": repeat,
            }
        )
        episode_index += 1
    return plan


def _episode_metrics(
    *,
    checkpoint: str,
    plan_item: dict[str, Any],
    length: int,
    total_reward: float,
    terminated: bool,
    truncated: bool,
    termination_reason: str,
    records: list[dict[str, float]],
    max_steps: int,
) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [_float(record.get(key, 0.0)) for record in records]

    termination_step = length if terminated else ""
    return {
        "checkpoint": checkpoint,
        "seed": plan_item["seed"],
        "vx": plan_item["vx"],
        "vz": plan_item["vz"],
        "yaw_rate": plan_item["yaw_rate"],
        "vx_bin": plan_item["vx_bin"],
        "vz_bin": plan_item["vz_bin"],
        "yaw_bin": plan_item["yaw_bin"],
        "bin_kind": plan_item["bin_kind"],
        "episode_index": plan_item["episode_index"],
        "episode_length": int(length),
        "survived_1000": int(length >= 1000 and not terminated),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "termination_reason": termination_reason,
        "termination_step": termination_step,
        "reward": float(total_reward),
        "mean_abs_vx_error": _mean([abs(v) for v in values("lin_vel_error_x")]),
        "mean_abs_vz_error": _mean([abs(v) for v in values("lin_vel_error_z")]),
        "mean_abs_yaw_error": _mean([abs(v) for v in values("yaw_rate_error")]),
        "mean_torque_fraction": _mean(values("mean_torque_limit_fraction")),
        "max_torque_fraction": float(max(values("max_torque_limit_fraction"), default=0.0)),
        "mean_abs_action": _mean(values("mean_abs_action")),
        "max_abs_action": float(max(values("max_abs_action"), default=0.0)),
        "nonfoot_contact_count": _mean(values("contact_diagnostic_count")),
        "nonfoot_contact_count_sum": float(sum(values("contact_diagnostic_count"))),
    }


def _record_from_info(info: dict[str, Any]) -> dict[str, float]:
    terms = info.get("reward_terms", {})
    return {
        "lin_vel_error_x": _float(terms.get("lin_vel_error_x")),
        "lin_vel_error_z": _float(terms.get("lin_vel_error_z")),
        "yaw_rate_error": _float(terms.get("yaw_rate_error")),
        "mean_torque_limit_fraction": _float(terms.get("mean_torque_limit_fraction")),
        "max_torque_limit_fraction": _float(terms.get("max_torque_limit_fraction")),
        "mean_abs_action": _float(terms.get("mean_abs_action")),
        "max_abs_action": _float(terms.get("max_abs_action")),
        "contact_diagnostic_count": _float(terms.get("contact_diagnostic_count")),
    }


def _send_error(conn: Any, exc: BaseException) -> None:
    try:
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    except (BrokenPipeError, EOFError, OSError):
        pass


def _env_worker(conn: Any, max_steps: int, seed_base: int, worker_id: int) -> None:
    _set_thread_env()
    env = None
    try:
        env = Go1SCMEnv(
            max_steps=max_steps,
            enable_motors=True,
            fixed_command=(0.0, 0.0, 0.0),
            command_seed=int(seed_base) + 1000 * int(worker_id),
            env_rank=int(worker_id),
            actuator_model="actuator_net",
            visual_mesh_format="none",
        )
        conn.send({"ok": True, "type": "ready"})

        while True:
            message = conn.recv()
            command = message.get("cmd")
            if command == "close":
                conn.send({"ok": True, "type": "closed"})
                break
            if command == "start_episode":
                item = message["item"]
                env.set_fixed_command(item["vx"], item["vz"], item["yaw_rate"])
                torch.manual_seed(int(item["seed"]))
                np.random.seed(int(item["seed"]) % (2**32 - 1))
                obs, _reset_info = env.reset(seed=int(item["seed"]))
                conn.send({"ok": True, "type": "reset", "obs": obs})
                continue
            if command == "step":
                obs, reward, terminated, truncated, info = env.step(np.asarray(message["action"], dtype=np.float32))
                conn.send(
                    {
                        "ok": True,
                        "type": "step",
                        "obs": obs,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "termination_reason": str(info.get("termination_reason") or ""),
                        "record": _record_from_info(info),
                    }
                )
                continue
            raise ValueError(f"Unknown worker command: {command}")
    except BaseException as exc:
        _send_error(conn, exc)
    finally:
        if env is not None:
            env.close()
        conn.close()


def _recv_ok(conn: Any, worker_id: int) -> dict[str, Any]:
    response = conn.recv()
    if not response.get("ok", False):
        raise RuntimeError(f"Worker {worker_id} failed: {response.get('error', 'unknown error')}")
    return response


def _close_workers(workers: list[tuple[int, Any, mp.Process]]) -> None:
    for worker_id, conn, process in workers:
        try:
            if process.is_alive():
                conn.send({"cmd": "close"})
                _recv_ok(conn, worker_id)
        except (BrokenPipeError, EOFError, OSError, RuntimeError):
            pass
        finally:
            conn.close()
    for _worker_id, _conn, process in workers:
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _start_env_workers(
    *,
    count: int,
    method: str,
    max_steps: int,
    seed_base: int,
) -> list[tuple[int, Any, mp.Process]]:
    ctx = mp.get_context(method)
    workers: list[tuple[int, Any, mp.Process]] = []
    try:
        for worker_id in range(count):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_env_worker,
                args=(child_conn, int(max_steps), int(seed_base), worker_id),
            )
            process.start()
            child_conn.close()
            workers.append((worker_id, parent_conn, process))
        for worker_id, conn, _process in workers:
            _recv_ok(conn, worker_id)
        return workers
    except BaseException:
        _close_workers(workers)
        raise


def run_checkpoint_parallel(
    policy_path: Path,
    checkpoint: str,
    plan: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    workers = max(1, int(args.num_workers))
    worker_count = min(workers, len(plan))
    model = load_ppo_same_shape_action_space(EnvClippedActionPPO, policy_path, env=None, device=args.device)

    def run_with_method(method: str) -> list[dict[str, Any]]:
        print(
            f"{checkpoint}: started {worker_count} env workers, {len(plan)} episodes",
            flush=True,
        )
        started_workers = _start_env_workers(
            count=worker_count,
            method=method,
            max_steps=int(args.max_steps),
            seed_base=int(args.seed_base),
        )
        rows: list[dict[str, Any]] = []
        slots: dict[int, dict[str, Any]] = {}
        next_episode = 0
        completed = 0
        next_progress = 25

        def start_episode(worker_id: int, conn: Any) -> bool:
            nonlocal next_episode
            if next_episode >= len(plan):
                slots.pop(worker_id, None)
                return False
            item = plan[next_episode]
            next_episode += 1
            conn.send({"cmd": "start_episode", "item": item})
            response = _recv_ok(conn, worker_id)
            slots[worker_id] = {
                "item": item,
                "obs": np.asarray(response["obs"], dtype=np.float32),
                "records": [],
                "total_reward": 0.0,
                "terminated": False,
                "truncated": False,
                "termination_reason": "",
            }
            return True

        try:
            for worker_id, conn, _process in started_workers:
                start_episode(worker_id, conn)

            while completed < len(plan):
                active = [(worker_id, conn) for worker_id, conn, _process in started_workers if worker_id in slots]
                if not active:
                    raise RuntimeError("No active workers remain before evaluation completed.")

                obs_batch = np.stack([slots[worker_id]["obs"] for worker_id, _conn in active]).astype(np.float32)
                actions, _state = model.predict(obs_batch, deterministic=bool(args.deterministic))
                actions = np.asarray(actions, dtype=np.float32).reshape((len(active), -1))

                for action_index, (worker_id, conn) in enumerate(active):
                    conn.send({"cmd": "step", "action": actions[action_index]})

                for worker_id, conn in active:
                    response = _recv_ok(conn, worker_id)
                    slot = slots[worker_id]
                    slot["obs"] = np.asarray(response["obs"], dtype=np.float32)
                    slot["records"].append(response["record"])
                    slot["total_reward"] += float(response["reward"])
                    slot["terminated"] = bool(response["terminated"])
                    slot["truncated"] = bool(response["truncated"])
                    slot["termination_reason"] = str(response["termination_reason"] or "")

                    done = (
                        bool(slot["terminated"])
                        or bool(slot["truncated"])
                        or len(slot["records"]) >= int(args.max_steps)
                    )
                    if not done:
                        continue

                    rows.append(
                        _episode_metrics(
                            checkpoint=checkpoint,
                            plan_item=slot["item"],
                            length=len(slot["records"]),
                            total_reward=float(slot["total_reward"]),
                            terminated=bool(slot["terminated"]),
                            truncated=bool(slot["truncated"]) or len(slot["records"]) >= int(args.max_steps),
                            termination_reason=str(slot["termination_reason"]),
                            records=slot["records"],
                            max_steps=int(args.max_steps),
                        )
                    )
                    completed += 1
                    slots.pop(worker_id, None)
                    if completed >= next_progress or completed == len(plan):
                        print(f"{checkpoint}: completed {completed}/{len(plan)} episodes", flush=True)
                        next_progress = ((completed // 25) + 1) * 25
                    start_episode(worker_id, conn)
        finally:
            _close_workers(started_workers)

        rows.sort(key=lambda row: int(row["episode_index"]))
        return rows

    try:
        return run_with_method(args.worker_start_method)
    except PermissionError:
        if args.worker_start_method != "forkserver":
            raise
        print("forkserver unavailable; falling back to fork for this run", flush=True)
        args.effective_worker_start_method = "fork"
        return run_with_method("fork")


def aggregate_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["vx_bin"], row["vz_bin"], row["yaw_bin"])].append(row)

    aggregates: list[dict[str, Any]] = []
    for (vx_bin, vz_bin, yaw_bin), items in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        lengths = [float(item["episode_length"]) for item in items]
        survived = [float(item["survived_1000"]) for item in items]
        reason_counts = Counter(str(item.get("termination_reason") or "none") for item in items)
        checkpoint = str(items[0]["checkpoint"])
        aggregates.append(
            {
                "checkpoint": checkpoint,
                "vx_bin": vx_bin,
                "vz_bin": vz_bin,
                "yaw_bin": yaw_bin,
                "episodes": len(items),
                "survival_rate": _mean(survived),
                "failure_rate": 1.0 - _mean(survived),
                "mean_episode_length": _mean(lengths),
                "median_episode_length": _median(lengths),
                "min_episode_length": float(min(lengths, default=0.0)),
                "mean_steps_lost": float(max(0.0, 1000.0 - _mean(lengths))),
                "mean_abs_vx_error": _mean([float(item["mean_abs_vx_error"]) for item in items]),
                "mean_abs_vz_error": _mean([float(item["mean_abs_vz_error"]) for item in items]),
                "mean_abs_yaw_error": _mean([float(item["mean_abs_yaw_error"]) for item in items]),
                "termination_reason_counts": json.dumps(dict(sorted(reason_counts.items())), sort_keys=True),
            }
        )
    return aggregates


def compare_bins(baseline: list[dict[str, Any]], latest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["vx_bin"], row["vz_bin"], row["yaw_bin"]): row
        for row in baseline
    }
    latest_by_key = {
        (row["vx_bin"], row["vz_bin"], row["yaw_bin"]): row
        for row in latest
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(by_key) & set(latest_by_key), key=lambda item: tuple(str(x) for x in item)):
        base = by_key[key]
        new = latest_by_key[key]
        rows.append(
            {
                "vx_bin": key[0],
                "vz_bin": key[1],
                "yaw_bin": key[2],
                "survival_rate_571m": base["survival_rate"],
                "survival_rate_latest": new["survival_rate"],
                "delta_survival_rate": float(new["survival_rate"]) - float(base["survival_rate"]),
                "mean_len_571m": base["mean_episode_length"],
                "mean_len_latest": new["mean_episode_length"],
                "delta_mean_len": float(new["mean_episode_length"]) - float(base["mean_episode_length"]),
                "mean_steps_lost_571m": base["mean_steps_lost"],
                "mean_steps_lost_latest": new["mean_steps_lost"],
                "delta_steps_lost": float(new["mean_steps_lost"]) - float(base["mean_steps_lost"]),
            }
        )
    return rows


def worst_bins(*aggregate_sets: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aggregates in aggregate_sets:
        ordered = sorted(
            aggregates,
            key=lambda row: (float(row["mean_steps_lost"]), float(row["failure_rate"])),
            reverse=True,
        )
        for rank, row in enumerate(ordered[:limit], start=1):
            rows.append({"rank": rank, **row})
    return rows


def main() -> None:
    args = parse_args()
    args.effective_worker_start_method = args.worker_start_method
    args.out.mkdir(parents=True, exist_ok=True)

    baseline_id = _checkpoint_id(args.baseline_policy)
    latest_id = _checkpoint_id(args.latest_policy)
    plan = build_episode_plan(args)

    baseline_rows = run_checkpoint_parallel(args.baseline_policy, baseline_id, plan, args)
    latest_rows = run_checkpoint_parallel(args.latest_policy, latest_id, plan, args)

    manifest = {
        "baseline_policy": str(args.baseline_policy),
        "latest_policy": str(args.latest_policy),
        "baseline_checkpoint": baseline_id,
        "latest_checkpoint": latest_id,
        "out": str(args.out),
        "episodes_per_bin": int(args.episodes_per_bin),
        "standing_episodes": int(args.standing_episodes),
        "max_steps": int(args.max_steps),
        "seed_base": int(args.seed_base),
        "command_seed": int(args.command_seed),
        "deterministic": bool(args.deterministic),
        "stochastic": not bool(args.deterministic),
        "vx_bins": VX_BINS,
        "vz_bins": VZ_BINS,
        "yaw_bins": YAW_BINS,
        "episode_count_per_checkpoint": len(plan),
        "parallel": int(args.num_workers) > 1,
        "parallel_mode": "batched_parent_policy_env_workers",
        "num_workers": int(args.num_workers),
        "worker_start_method": args.worker_start_method,
        "effective_worker_start_method": args.effective_worker_start_method,
        "model_copies_per_checkpoint": 1,
        "envs_per_checkpoint": min(max(1, int(args.num_workers)), len(plan)),
        "scm": scm_env_metadata(),
    }
    (args.out / "manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2) + "\n", encoding="utf-8")

    baseline_bins = aggregate_bins(baseline_rows)
    latest_bins = aggregate_bins(latest_rows)
    comparison = compare_bins(baseline_bins, latest_bins)
    worst = worst_bins(baseline_bins, latest_bins)

    _write_csv(args.out / f"episodes_{baseline_id}.csv", baseline_rows)
    _write_csv(args.out / f"episodes_{latest_id}.csv", latest_rows)
    _write_csv(args.out / f"bins_{baseline_id}.csv", baseline_bins)
    _write_csv(args.out / f"bins_{latest_id}.csv", latest_bins)
    _write_csv(args.out / "comparison_bins.csv", comparison)
    _write_csv(args.out / "worst_bins_by_checkpoint.csv", worst)

    summary = {
        "out": str(args.out),
        "baseline_checkpoint": baseline_id,
        "latest_checkpoint": latest_id,
        "episodes_per_checkpoint": len(plan),
        "files": [
            f"episodes_{baseline_id}.csv",
            f"episodes_{latest_id}.csv",
            f"bins_{baseline_id}.csv",
            f"bins_{latest_id}.csv",
            "comparison_bins.csv",
            "worst_bins_by_checkpoint.csv",
            "manifest.json",
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
