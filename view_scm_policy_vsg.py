"""View the default Go1 policy on SCM deformable terrain with Chrono VSG."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np


def _import_chrono_vsg():
    try:
        import pychrono as chrono
        import pychrono.vsg3d as vsg
    except Exception as exc:
        raise RuntimeError(
            "SCM VSG viewer requires the source-built Chrono environment. "
            "Run it from chrono-src with:\n"
            '  export LD_LIBRARY_PATH="$HOME/chrono_builds/chrono-install/lib:'
            '$HOME/chrono_builds/vsg-install/lib:/usr/lib/wsl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"\n'
            '  export PYTHONPATH="$HOME/chrono_builds/chrono-install/share/chrono/python:$PYTHONPATH"\n'
            "Missing import while loading pychrono.vsg3d."
        ) from exc

    missing = []
    if not hasattr(vsg, "ChVisualSystemVSG"):
        missing.append("pychrono.vsg3d.ChVisualSystemVSG")
    if missing:
        raise RuntimeError(
            "SCM VSG viewer requires a Chrono build with VSG support. "
            f"Missing: {', '.join(missing)}"
        )
    return chrono, vsg


chrono, vsg = _import_chrono_vsg()

from diagnostics import (  # noqa: E402
    contact_body_groups,
    contact_debug_stats,
    foot_bodies,
    foot_debug_stats,
    foot_xz_positions,
    format_foot_values,
    new_interval_stats,
    update_interval_stats,
)
from ppo_compat import EnvClippedActionPPO, load_ppo_same_shape_action_space  # noqa: E402
from project_config import CURRENT_BASELINE_MODEL, SB3_DEVICE  # noqa: E402
from go1_scm_env import Go1SCMEnv  # noqa: E402


ACTUATOR_MODELS = ("actuator_net", "torque_limited_pd")
VISUAL_MESH_FORMATS = ("obj", "obj_lod50", "urdf", "none")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy",
        type=Path,
        nargs="?",
        default=CURRENT_BASELINE_MODEL,
        help="Path to a Stable-Baselines3 policy zip.",
    )
    parser.add_argument("--fixed-command-vx", type=float, default=0.0)
    parser.add_argument("--fixed-command-vz", type=float, default=0.0)
    parser.add_argument("--fixed-command-yaw-rate", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--actuator-model", choices=ACTUATOR_MODELS, default="actuator_net")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--zero-action",
        action="store_true",
        help="Bypass the policy and send literal all-zero actions every step.",
    )
    parser.add_argument("--ignore-termination", action="store_true")
    parser.add_argument("--full-diagnostics", action="store_true")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--render-fps", type=float, default=50.0)
    parser.add_argument("--visual-mesh-format", choices=VISUAL_MESH_FORMATS, default="obj")
    parser.add_argument("--show-collision-boxes", action="store_true")
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.0)
    parser.add_argument("--spawn-height", type=float, default=None)
    parser.add_argument("--disable-dynamic-spawn-height", action="store_true")
    parser.add_argument("--follow-camera", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=6.0)
    parser.add_argument("--camera-height", type=float, default=3.0)
    parser.add_argument("--camera-target-height", type=float, default=0.30)
    parser.add_argument("--camera-lead", type=float, default=0.0)
    parser.add_argument("--camera-yaw-deg", type=float, default=45.0)
    parser.add_argument("--debug-startup", action="store_true")
    return parser.parse_args()


def _term(reward_terms: dict, name: str) -> float:
    return float(reward_terms.get(name, 0.0))


def _max(values) -> float:
    return float(max(values)) if values else 0.0


def _trace(args, message: str) -> None:
    if args.debug_startup:
        print(f"[scm-vsg] {message}", file=sys.stderr, flush=True)


def camera_vectors(env, args):
    if env._trunk is None:
        return chrono.ChVector3d(-1.0, 0.65, 3.0), chrono.ChVector3d(0.0, 0.3, 0.0)
    pos = env._trunk.GetPos()
    yaw = np.deg2rad(float(args.camera_yaw_deg))
    camera = chrono.ChVector3d(
        float(pos.x) - float(args.camera_lead) + float(args.camera_distance) * float(np.cos(yaw)),
        float(args.camera_height),
        float(pos.z) + float(args.camera_distance) * float(np.sin(yaw)),
    )
    target = chrono.ChVector3d(
        float(pos.x) - float(args.camera_lead),
        float(pos.y) + float(args.camera_target_height),
        float(pos.z),
    )
    return camera, target


def make_vsg_visualizer(env, args):
    _trace(args, "constructing ChVisualSystemVSG")
    vis = vsg.ChVisualSystemVSG()
    _trace(args, "AttachSystem(system)")
    vis.AttachSystem(env._system)
    _trace(args, "SetWindowTitle")
    vis.SetWindowTitle("Go1 SCM Policy Viewer")
    _trace(args, "SetWindowSize")
    vis.SetWindowSize(1280, 720)
    if hasattr(vis, "SetCameraVertical") and hasattr(chrono, "CameraVerticalDir_Y"):
        _trace(args, "SetCameraVertical(Y)")
        vis.SetCameraVertical(chrono.CameraVerticalDir_Y)
    if hasattr(vis, "EnableSkyBox"):
        _trace(args, "EnableSkyBox")
        vis.EnableSkyBox()
    if hasattr(vis, "SetBackgroundColor"):
        _trace(args, "SetBackgroundColor")
        vis.SetBackgroundColor(chrono.ChColor(0.08, 0.09, 0.10))
    if hasattr(vis, "SetLightDirection"):
        _trace(args, "SetLightDirection")
        vis.SetLightDirection(-0.6, 0.8)
    if hasattr(vis, "SetLightIntensity"):
        _trace(args, "SetLightIntensity")
        vis.SetLightIntensity(1.0)
    camera, target = camera_vectors(env, args)
    if hasattr(vis, "AddCamera"):
        _trace(args, "AddCamera")
        vis.AddCamera(camera, target)
    _trace(args, "Initialize")
    vis.Initialize()
    _trace(args, "Initialize done")
    update_follow_camera(vis, env, args)
    return vis


def update_follow_camera(vis, env, args) -> None:
    if not args.follow_camera or env._trunk is None:
        return
    camera, target = camera_vectors(env, args)
    vis.SetCameraPosition(camera)
    vis.SetCameraTarget(target)


def print_compact_policy_step(episode: int, step: int, info: dict, foot_stats: dict, interval_stats: dict) -> None:
    terms = info.get("reward_terms", {})
    foot_load_min = min(interval_stats["foot_load_min"])
    print(
        f"ep={episode:03d} step={step:04d} "
        f"reward={_term(terms, 'reward_dt_scaled'):+.3f} "
        f"cmd=({_term(terms, 'command_vx'):+.2f},{_term(terms, 'command_vz'):+.2f},{_term(terms, 'command_yaw_rate'):+.2f}) "
        f"body=({_term(terms, 'body_lin_vel_x'):+.3f},{_term(terms, 'body_lin_vel_z'):+.3f},{_term(terms, 'body_yaw_rate'):+.3f}) "
        f"act={_term(terms, 'mean_abs_action'):.3f} "
        f"torque={_term(terms, 'mean_abs_motor_torque'):.2f}Nm "
        f"sat={_term(terms, 'fraction_torque_saturated'):.3f} "
        f"foot_min={foot_load_min:.1f}N "
        f"load_imb={foot_stats['foot_load_imbalance']:.2f} "
        f"slip={foot_stats['foot_dxz_max']:.4f}m "
        f"nonfoot_max={_max(interval_stats['nonfoot_load_max']):.1f}N"
    )


def print_full_policy_step(
    episode: int,
    step: int,
    action: np.ndarray,
    info: dict,
    foot_stats: dict,
    contact_stats: dict,
    interval_stats: dict,
) -> None:
    terms = info.get("reward_terms", {})
    group_loads = contact_stats["group_loads"]
    print(
        f"ep={episode:03d} step={step:04d} "
        f"reward={_term(terms, 'reward_dt_scaled'):+.4f} raw={_term(terms, 'reward_raw_sum'):+.3f} "
        f"track=({_term(terms, 'tracking_lin_vel_reward'):+.3f},{_term(terms, 'tracking_ang_vel_reward'):+.3f}) "
        f"orient={_term(terms, 'flat_orientation_l2_reward'):+.3f} "
        f"rate={_term(terms, 'action_rate_reward'):+.3f} "
        f"torque={_term(terms, 'torques_reward'):+.3f} "
        f"cmd=({_term(terms, 'command_vx'):+.2f},{_term(terms, 'command_vz'):+.2f},{_term(terms, 'command_yaw_rate'):+.2f}) "
        f"body=({_term(terms, 'body_lin_vel_x'):+.3f},{_term(terms, 'body_lin_vel_z'):+.3f},{_term(terms, 'body_yaw_rate'):+.3f}) "
        f"act_mean={float(np.mean(np.abs(action))):.3f}"
        f" foot_dxz_mean={foot_stats['foot_dxz_mean']:.4f}"
        f" foot_dxz_max={foot_stats['foot_dxz_max']:.4f}"
        f" load_imb={foot_stats['foot_load_imbalance']:.3f}"
        f" {format_foot_values('foot_load', group_loads['foot'], 1)}"
        f" {format_foot_values('calf_load', group_loads['calf'], 1)}"
        f" {format_foot_values('thigh_load', group_loads['thigh'], 1)}"
        f" {format_foot_values('foot_load_min', interval_stats['foot_load_min'], 1)}"
        f" {format_foot_values('foot_load_max', interval_stats['foot_load_max'], 1)}"
        f" {format_foot_values('nonfoot_load_max', interval_stats['nonfoot_load_max'], 1)}"
    )


def main() -> None:
    args = parse_args()
    if args.render_fps <= 0:
        raise ValueError("--render-fps must be positive")
    if not args.zero_action and not args.policy.exists():
        raise FileNotFoundError(f"Policy not found: {args.policy}")

    env = Go1SCMEnv(
        render_mode=None,
        max_steps=args.max_steps,
        enable_motors=True,
        spawn_x=args.spawn_x,
        spawn_z=args.spawn_z,
        **({} if args.spawn_height is None else {"spawn_height": args.spawn_height}),
        fixed_command=(args.fixed_command_vx, args.fixed_command_vz, args.fixed_command_yaw_rate),
        command_seed=1,
        actuator_model=args.actuator_model,
        default_randomization=not args.zero_action,
        observation_noise=not args.zero_action,
        dynamic_spawn_height=not args.disable_dynamic_spawn_height,
        show_collision_shapes=args.show_collision_boxes,
        visual_mesh_format=args.visual_mesh_format,
    )
    model = None
    if not args.zero_action:
        model = load_ppo_same_shape_action_space(EnvClippedActionPPO, args.policy, env=env, device=SB3_DEVICE)

    vis = None
    obs = None
    tracked_feet = None
    tracked_contacts = None
    reset_foot_xz = None
    interval_stats = new_interval_stats()

    def reset_episode():
        nonlocal vis, obs, tracked_feet, tracked_contacts, reset_foot_xz, interval_stats
        _trace(args, "env.reset")
        obs, reset_info = env.reset()
        _trace(args, "env.reset done")
        tracked_feet = foot_bodies(env)
        tracked_contacts = contact_body_groups(env)
        reset_foot_xz = foot_xz_positions(tracked_feet)
        interval_stats = new_interval_stats()
        vis = make_vsg_visualizer(env, args)
        return reset_info

    reset_info = reset_episode()
    print(
        f"viewing SCM policy={'zero_action_bypass' if args.zero_action else args.policy} actuator={args.actuator_model} "
        f"fixed=({args.fixed_command_vx:.2f},{args.fixed_command_vz:.2f},{args.fixed_command_yaw_rate:.2f}) "
        f"stochastic={args.stochastic} visual_mesh_format={args.visual_mesh_format} "
        f"show_collision_boxes={args.show_collision_boxes} render_fps={args.render_fps:.1f}"
    )
    print(f"command_sampler={reset_info.get('command_sampler')}")
    print(f"scm={reset_info.get('scm')}")

    step = 0
    episode = 1
    render_interval = 1.0 / float(args.render_fps)
    next_render_time = 0.0
    try:
        while vis is not None and vis.Run():
            sim_time = float(env.step_count) * 0.02
            if sim_time + 1e-12 >= next_render_time:
                update_follow_camera(vis, env, args)
                if hasattr(vis, "BeginScene"):
                    vis.BeginScene()
                vis.Render()
                if hasattr(vis, "EndScene"):
                    vis.EndScene()
                while next_render_time <= sim_time + 1e-12:
                    next_render_time += render_interval
            if args.zero_action:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _reward, terminated, truncated, info = env.step(action)

            foot_stats = foot_debug_stats(tracked_feet, reset_foot_xz)
            contact_stats = contact_debug_stats(tracked_contacts)
            update_interval_stats(interval_stats, foot_stats, contact_stats)
            if args.log_interval > 0 and step % args.log_interval == 0:
                if args.full_diagnostics:
                    print_full_policy_step(
                        episode,
                        step,
                        np.asarray(action),
                        info,
                        foot_stats,
                        contact_stats,
                        interval_stats,
                    )
                else:
                    print_compact_policy_step(episode, step, info, foot_stats, interval_stats)
                interval_stats = new_interval_stats()

            step += 1
            termination_reason = info.get("termination_reason")
            invalid_state = termination_reason == "invalid_obs" or not np.all(np.isfinite(obs))
            ended = truncated or invalid_state or (terminated and not args.ignore_termination)
            if ended:
                print(f"ep={episode:03d} ended reason={termination_reason or 'truncated'} steps={step}")
                reset_info = reset_episode()
                step = 0
                next_render_time = 0.0
                episode += 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
