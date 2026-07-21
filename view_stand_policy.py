"""View a trained Chrono Go1 standing policy in Irrlicht."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

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
from go1_env import Go1Env, _HOME_JOINT_ANGLES, _JOINT_NAMES
from project_config import (
    CURRENT_BASELINE_MODEL,
    DEFAULT_VIEWER_FRICTION_RANGE,
    SB3_DEVICE,
)


RESET_NOISE_LEVELS = ("clean", "rn1", "rn2", "rn3")
RESET_NOISE_COMPONENTS = ("combined", "joint_pos", "joint_vel", "roll_pitch", "yaw", "base_height", "base_position", "base_velocity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy",
        type=Path,
        nargs="?",
        default=CURRENT_BASELINE_MODEL,
        help="Path to a Stable-Baselines3 policy zip.",
    )
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=DEFAULT_VIEWER_FRICTION_RANGE[0])
    parser.add_argument("--friction-max", type=float, default=DEFAULT_VIEWER_FRICTION_RANGE[1])
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.0)
    parser.add_argument("--ground-height-offset", type=float, default=0.0)
    parser.add_argument("--reset-noise-level", choices=RESET_NOISE_LEVELS, default="clean")
    parser.add_argument("--reset-noise-components", choices=RESET_NOISE_COMPONENTS, default="combined")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print one compact status line every N steps.",
    )
    parser.add_argument(
        "--joint-debug",
        action="store_true",
        help="Print reset joint angle errors for pose observation debugging.",
    )
    parser.add_argument(
        "--full-diagnostics",
        action="store_true",
        help="Print the full per-foot contact/debug fields instead of the compact status line.",
    )
    parser.add_argument(
        "--ignore-termination",
        action="store_true",
        help="Keep stepping after height/tip termination so a fall can be watched.",
    )
    parser.add_argument(
        "--no-reset-on-end",
        action="store_true",
        help="Close the viewer instead of rebuilding the sim when an episode ends.",
    )
    return parser.parse_args()


def print_joint_debug(obs) -> None:
    joint_pos = obs[11:23]
    joint_error = joint_pos - _HOME_JOINT_ANGLES
    print("joint_debug step=reset")
    for name, measured, home, error in zip(
        _JOINT_NAMES,
        joint_pos,
        _HOME_JOINT_ANGLES,
        joint_error,
    ):
        print(
            f"  {name:<15} measured={float(measured): .4f} "
            f"home={float(home): .4f} error={float(error): .4f}"
        )
    print(f"  mean_squared_error={float((joint_error ** 2).mean()):.6f}")


def _term(reward_terms: dict, name: str) -> float:
    return float(reward_terms.get(name, 0.0))


def _max(values) -> float:
    return float(max(values)) if values else 0.0


def print_compact_policy_step(
    episode: int,
    step: int,
    info: dict,
    foot_stats: dict,
    interval_stats: dict,
) -> None:
    terms = info.get("reward_terms", {})
    foot_load_min = min(interval_stats["foot_load_min"])
    print(
        f"ep={episode:03d} step={step:04d} "
        f"h_rel={_term(terms, 'base_relative_height'):.3f} "
        f"up={_term(terms, 'upright_score'):.3f} "
        f"xz=({_term(terms, 'lin_vel_x'):+.3f},{_term(terms, 'lin_vel_z'):+.3f}) "
        f"act={_term(terms, 'mean_abs_action'):.3f} "
        f"dact={_term(terms, 'mean_abs_action_delta'):.3f} "
        f"jvel={_term(terms, 'mean_abs_joint_vel'):.3f} "
        f"tilt={_term(terms, 'tilt_error'):.4f} "
        f"foot_min={foot_load_min:.1f}N "
        f"load_imb={foot_stats['foot_load_imbalance']:.2f} "
        f"slip={foot_stats['foot_dxz_max']:.4f}m "
        f"vfoot={foot_stats['foot_vxz_max']:.4f}m/s "
        f"nonfoot_max={_max(interval_stats['nonfoot_load_max']):.1f}N"
    )


def print_full_policy_step(
    episode: int,
    step: int,
    action,
    obs,
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
            f" foot_vxz_mean={foot_stats['foot_vxz_mean']:.4f}"
            f" foot_vxz_max={foot_stats['foot_vxz_max']:.4f}"
            f" load_imb={foot_stats['foot_load_imbalance']:.3f}"
            f" {format_foot_values('foot_y', foot_stats['foot_heights'])}"
            f" {format_foot_values('foot_share', foot_stats['foot_load_shares'], 2)}"
            f" {format_foot_values('foot_dxz', foot_stats['foot_displacements'])}"
        )
    contact_text = ""
    if contact_stats is not None:
        group_loads = contact_stats["group_loads"]
        contact_text = (
            f" {format_foot_values('foot_load', group_loads['foot'], 1)}"
            f" {format_foot_values('calf_load', group_loads['calf'], 1)}"
            f" {format_foot_values('thigh_load', group_loads['thigh'], 1)}"
            f" {format_foot_values('hip_load', group_loads['hip'], 1)}"
            f" {format_foot_values('leg_nonfoot_load', contact_stats['nonfoot_loads'], 1)}"
        )
    interval_text = ""
    if interval_stats is not None:
        interval_text = (
            f" {format_foot_values('foot_y_min', interval_stats['foot_y_min'])}"
            f" {format_foot_values('foot_y_max', interval_stats['foot_y_max'])}"
            f" {format_foot_values('foot_load_min', interval_stats['foot_load_min'], 1)}"
            f" {format_foot_values('foot_load_max', interval_stats['foot_load_max'], 1)}"
            f" {format_foot_values('nonfoot_load_max', interval_stats['nonfoot_load_max'], 1)}"
        )
    print(
        f"ep={episode:03d} step={step:04d} "
        f"height_rel={_term(terms, 'base_relative_height'):.3f} "
        f"upright={_term(terms, 'upright_score'):.3f} "
        f"xz_vel=({_term(terms, 'lin_vel_x'):+.3f},"
        f"{_term(terms, 'lin_vel_z'):+.3f}) "
        f"act_mean={_term(terms, 'mean_abs_action'):.3f} "
        f"dact_mean={_term(terms, 'mean_abs_action_delta'):.3f} "
        f"jvel_mean={_term(terms, 'mean_abs_joint_vel'):.3f} "
        f"sym={_term(terms, 'leg_symmetry_error'):.4f} "
        f"tilt={_term(terms, 'tilt_error'):.4f} "
        f"pose_err={_term(terms, 'pose_error'):.4f} "
        f"{foot_text}"
        f"{contact_text}"
        f"{interval_text}"
    )


def main() -> None:
    args = parse_args()
    if not args.policy.exists():
        raise FileNotFoundError(
            f"Policy not found: {args.policy}. Pass the model zip path, for "
            "example: view_stand_policy.py runs/stand/final_model.zip"
        )

    env = Go1Env(
        render_mode="human",
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=True,
        friction_range=(args.friction_min, args.friction_max),
        reset_noise_level=args.reset_noise_level,
        reset_noise_components=args.reset_noise_components,
        spawn_x=args.spawn_x,
        spawn_z=args.spawn_z,
        ground_height_offset=args.ground_height_offset,
    )
    model = PPO.load(args.policy, device=SB3_DEVICE)
    obs, reset_info = env.reset()
    tracked_feet = foot_bodies(env)
    tracked_contacts = contact_body_groups(env)
    reset_foot_xz = foot_xz_positions(tracked_feet)
    interval_stats = new_interval_stats()
    if args.joint_debug:
        print_joint_debug(obs)
    print(
        f"viewing policy={args.policy} terrain={args.terrain} "
        f"friction=({args.friction_min:.2f}, {args.friction_max:.2f}) "
        f"spawn=({args.spawn_x:.2f}, {args.spawn_z:.2f}) "
        f"ground_height_offset={args.ground_height_offset:.2f} "
        f"reset_noise={args.reset_noise_level}/{args.reset_noise_components}"
    )
    print(f"reset_noise_sample={reset_info['reset_noise']}")
    step = 0
    episode = 1

    try:
        while env.render():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            foot_stats = foot_debug_stats(tracked_feet, reset_foot_xz)
            contact_stats = contact_debug_stats(tracked_contacts)
            update_interval_stats(interval_stats, foot_stats, contact_stats)
            if args.log_interval > 0 and step % args.log_interval == 0:
                if args.full_diagnostics:
                    print_full_policy_step(
                        episode,
                        step,
                        action,
                        obs,
                        info,
                        foot_stats,
                        contact_stats,
                        interval_stats,
                    )
                else:
                    print_compact_policy_step(
                        episode,
                        step,
                        info,
                        foot_stats,
                        interval_stats,
                    )
                interval_stats = new_interval_stats()
            step += 1
            ended = truncated or (terminated and not args.ignore_termination)
            if ended:
                print(
                    f"ep={episode:03d} ended "
                    f"reason={info.get('termination_reason') or 'truncated'} "
                    f"steps={step}"
                )
                if args.no_reset_on_end:
                    break
                obs, reset_info = env.reset()
                print(f"reset_noise_sample={reset_info['reset_noise']}")
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
