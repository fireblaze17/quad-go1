"""View the default Chrono Go1 policy in Irrlicht."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pychrono as chrono

from diagnostics import (
    contact_body_groups,
    contact_debug_stats,
    foot_bodies,
    foot_debug_stats,
    foot_xz_positions,
    format_foot_values,
    new_interval_stats,
    update_interval_stats,
)
from go1_env import Go1Env, _HOME_JOINT_ANGLES, _JOINT_NAMES, _TIME_STEP
from ppo_compat import EnvClippedActionPPO, load_ppo_same_shape_action_space
from project_config import CURRENT_BASELINE_MODEL, SB3_DEVICE


ACTUATOR_MODELS = ("actuator_net", "torque_limited_pd")
ENV_BACKENDS = ("flat", "scm")
VISUAL_MESH_FORMATS = ("obj", "obj_lod50", "urdf", "none")


def env_class_for_backend(env_backend: str):
    if env_backend == "scm":
        from go1_scm_env import Go1SCMEnv

        return Go1SCMEnv
    if env_backend == "flat":
        return Go1Env
    raise ValueError(f"unsupported env backend: {env_backend}")


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
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--env-backend", choices=ENV_BACKENDS, default="flat")
    parser.add_argument("--actuator-model", choices=ACTUATOR_MODELS, default="actuator_net")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--ignore-termination", action="store_true")
    parser.add_argument("--full-diagnostics", action="store_true")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--joint-debug", action="store_true")
    parser.add_argument("--no-reset-on-end", action="store_true")
    parser.add_argument("--visual-mesh-format", choices=VISUAL_MESH_FORMATS, default="obj")
    parser.add_argument("--show-collision-boxes", action="store_true")
    parser.add_argument("--show-urdf-collision-visuals", action="store_true")
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.0)
    parser.add_argument("--spawn-height", type=float, default=None)
    parser.add_argument("--disable-dynamic-spawn-height", action="store_true")
    parser.add_argument("--no-follow-camera", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-height", type=float, default=0.65)
    parser.add_argument("--camera-target-height", type=float, default=0.30)
    parser.add_argument("--camera-lead", type=float, default=0.8)
    parser.add_argument("--camera-yaw-deg", type=float, default=90.0)
    parser.add_argument(
        "--viewer-command-sequence",
        choices=("none", "forward_then_backward", "backward_then_forward"),
        default="none",
    )
    parser.add_argument("--viewer-command-switch-step", type=int, default=500)
    parser.add_argument("--viewer-sequence-forward-vx", type=float, default=-1.0)
    parser.add_argument("--viewer-sequence-backward-vx", type=float, default=1.0)
    parser.add_argument("--viewer-push-step", type=int, default=-1)
    parser.add_argument("--viewer-push-duration-steps", type=int, default=1)
    parser.add_argument("--viewer-push-vx", type=float, default=0.0)
    parser.add_argument("--viewer-push-vz", type=float, default=0.0)
    return parser.parse_args()


def update_follow_camera(env: Go1Env, args) -> None:
    if args.no_follow_camera or env._vis is None or env._trunk is None:
        return
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
    env._vis.SetCameraPosition(camera)
    env._vis.SetCameraTarget(target)


def print_joint_debug(obs: np.ndarray) -> None:
    joint_pos = obs[12:24] + _HOME_JOINT_ANGLES
    joint_error = joint_pos - _HOME_JOINT_ANGLES
    print("joint_debug step=reset")
    for name, measured, home, error in zip(_JOINT_NAMES, joint_pos, _HOME_JOINT_ANGLES, joint_error):
        print(f"  {name:<15} measured={float(measured): .4f} home={float(home): .4f} error={float(error): .4f}")
    print(f"  mean_squared_error={float((joint_error ** 2).mean()):.6f}")


def apply_viewer_push(env: Go1Env, args, step: int) -> bool:
    if args.viewer_push_step < 0:
        return False
    duration = max(1, int(args.viewer_push_duration_steps))
    if not (args.viewer_push_step <= step < args.viewer_push_step + duration):
        return False
    trunk = getattr(env, "_trunk", None)
    if trunk is None:
        return False
    current_velocity = trunk.GetPosDt()
    trunk.SetPosDt(chrono.ChVector3d(float(args.viewer_push_vx), float(current_velocity.y), float(args.viewer_push_vz)))
    return True


def apply_viewer_command_sequence(env: Go1Env, args, step: int) -> bool:
    if args.viewer_command_sequence == "none":
        return False
    switch_step = max(0, int(args.viewer_command_switch_step))
    forward = float(args.viewer_sequence_forward_vx)
    backward = float(args.viewer_sequence_backward_vx)
    if args.viewer_command_sequence == "forward_then_backward":
        vx = forward if step < switch_step else backward
    else:
        vx = backward if step < switch_step else forward
    env.set_fixed_command(vx, 0.0, 0.0)
    return True


def _term(reward_terms: dict, name: str) -> float:
    return float(reward_terms.get(name, 0.0))


def _max(values) -> float:
    return float(max(values)) if values else 0.0


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
    foot_stats: dict | None = None,
    contact_stats: dict | None = None,
    interval_stats: dict | None = None,
) -> None:
    terms = info.get("reward_terms", {})
    foot_text = ""
    if foot_stats is not None:
        foot_text = (
            f" foot_dxz_mean={foot_stats['foot_dxz_mean']:.4f}"
            f" foot_dxz_max={foot_stats['foot_dxz_max']:.4f}"
            f" load_imb={foot_stats['foot_load_imbalance']:.3f}"
            f" {format_foot_values('foot_y', foot_stats['foot_heights'])}"
            f" {format_foot_values('foot_share', foot_stats['foot_load_shares'], 2)}"
            f" {format_foot_values('fric_use', foot_stats['foot_friction_usage'], 2)}"
        )
    contact_text = ""
    if contact_stats is not None:
        group_loads = contact_stats["group_loads"]
        contact_text = (
            f" {format_foot_values('foot_load', group_loads['foot'], 1)}"
            f" {format_foot_values('calf_load', group_loads['calf'], 1)}"
            f" {format_foot_values('thigh_load', group_loads['thigh'], 1)}"
        )
    interval_text = ""
    if interval_stats is not None:
        interval_text = (
            f" {format_foot_values('foot_load_min', interval_stats['foot_load_min'], 1)}"
            f" {format_foot_values('foot_load_max', interval_stats['foot_load_max'], 1)}"
            f" {format_foot_values('nonfoot_load_max', interval_stats['nonfoot_load_max'], 1)}"
        )
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
        f"{foot_text}{contact_text}{interval_text}"
    )


def main() -> None:
    args = parse_args()
    if not args.policy.exists():
        raise FileNotFoundError(f"Policy not found: {args.policy}")

    env_cls = env_class_for_backend(args.env_backend)
    env = env_cls(
        render_mode="human",
        max_steps=args.max_steps,
        enable_motors=True,
        spawn_x=args.spawn_x,
        spawn_z=args.spawn_z,
        **({} if args.spawn_height is None else {"spawn_height": args.spawn_height}),
        fixed_command=(args.fixed_command_vx, args.fixed_command_vz, args.fixed_command_yaw_rate),
        command_seed=1,
        actuator_model=args.actuator_model,
        dynamic_spawn_height=not args.disable_dynamic_spawn_height,
        show_collision_shapes=args.show_collision_boxes,
        show_urdf_collision_visuals=args.show_urdf_collision_visuals,
        visual_mesh_format=args.visual_mesh_format,
    )
    model = load_ppo_same_shape_action_space(EnvClippedActionPPO, args.policy, env=env, device=SB3_DEVICE)
    obs, reset_info = env.reset()
    tracked_feet = foot_bodies(env)
    tracked_contacts = contact_body_groups(env)
    reset_foot_xz = foot_xz_positions(tracked_feet)
    interval_stats = new_interval_stats()
    if args.joint_debug:
        print_joint_debug(obs)
    print(
        f"viewing policy={args.policy} actuator={args.actuator_model} "
        f"backend={args.env_backend} "
        f"fixed=({args.fixed_command_vx:.2f},{args.fixed_command_vz:.2f},{args.fixed_command_yaw_rate:.2f}) "
        f"stochastic={args.stochastic} visual_mesh_format={args.visual_mesh_format} "
        f"show_collision_boxes={args.show_collision_boxes}"
    )
    print(f"command_sampler={reset_info.get('command_sampler')}")
    print(f"default_randomization={reset_info.get('default_randomization')}")

    step = 0
    episode = 1
    try:
        while True:
            update_follow_camera(env, args)
            if not env.render():
                break
            apply_viewer_command_sequence(env, args, step)
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            if apply_viewer_push(env, args, step) and args.log_interval >= 0:
                print(f"viewer_push step={step} vx={args.viewer_push_vx:+.3f} vz={args.viewer_push_vz:+.3f}")
            obs, _reward, terminated, truncated, info = env.step(action)
            foot_stats = foot_debug_stats(tracked_feet, reset_foot_xz)
            contact_stats = contact_debug_stats(tracked_contacts)
            update_interval_stats(interval_stats, foot_stats, contact_stats)
            if args.log_interval > 0 and step % args.log_interval == 0:
                if args.full_diagnostics:
                    print_full_policy_step(episode, step, np.asarray(action), info, foot_stats, contact_stats, interval_stats)
                else:
                    print_compact_policy_step(episode, step, info, foot_stats, interval_stats)
                interval_stats = new_interval_stats()
            step += 1
            ended = truncated or (terminated and not args.ignore_termination)
            if ended:
                print(f"ep={episode:03d} ended reason={info.get('termination_reason') or 'truncated'} steps={step}")
                if args.no_reset_on_end:
                    break
                obs, reset_info = env.reset()
                tracked_feet = foot_bodies(env)
                tracked_contacts = contact_body_groups(env)
                reset_foot_xz = foot_xz_positions(tracked_feet)
                interval_stats = new_interval_stats()
                if args.joint_debug:
                    print_joint_debug(obs)
                step = 0
                episode += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
