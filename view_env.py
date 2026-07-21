"""Run the Chrono Go1 Gymnasium environment with the Irrlicht viewer."""

import argparse

import numpy as np

from diagnostics import FOOT_BODY_NAMES, foot_bodies, foot_debug_stats, foot_xz_positions
from go1_env import Go1Env


RESET_NOISE_LEVELS = ("clean", "rn1", "rn2", "rn3")
RESET_NOISE_COMPONENTS = ("combined", "joint_pos", "joint_vel", "roll_pitch", "yaw", "base_height", "base_position", "base_velocity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.8)
    parser.add_argument("--friction-max", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.0)
    parser.add_argument("--ground-height-offset", type=float, default=0.0)
    parser.add_argument("--reset-noise-level", choices=RESET_NOISE_LEVELS, default="clean")
    parser.add_argument("--reset-noise-components", choices=RESET_NOISE_COMPONENTS, default="combined")
    parser.add_argument("--disable-motors", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
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


def print_step(
    episode: int,
    step: int,
    action: np.ndarray,
    info: dict,
    foot_stats: dict,
) -> None:
    terms = info.get("reward_terms", {})
    print(
        f"ep={episode:03d} step={step:04d} "
        f"h_rel={float(terms.get('base_relative_height', 0.0)):.3f} "
        f"upright={float(terms.get('upright_score', 0.0)):.3f} "
        f"axis=({float(terms.get('trunk_x_up', 0.0)):+.3f},"
        f"{float(terms.get('trunk_y_up', 0.0)):+.3f},"
        f"{float(terms.get('trunk_z_up', 0.0)):+.3f}) "
        f"ang_xyz=({float(terms.get('ang_vel_x', 0.0)):+.3f},"
        f"{float(terms.get('ang_vel_y', 0.0)):+.3f},"
        f"{float(terms.get('ang_vel_z', 0.0)):+.3f}) "
        f"xz_vel=({float(terms.get('lin_vel_x', 0.0)):+.3f},"
        f"{float(terms.get('lin_vel_z', 0.0)):+.3f}) "
        f"pose_err={float(terms.get('pose_error', 0.0)):.4f} "
        f"jvel_mean={float(terms.get('mean_abs_joint_vel', 0.0)):.3f} "
        f"jvel_max={float(terms.get('max_abs_joint_vel', 0.0)):.3f} "
        f"dact_mean={float(terms.get('mean_abs_action_delta', 0.0)):.3f} "
        f"dact_max={float(terms.get('max_abs_action_delta', 0.0)):.3f} "
        f"amax={float(np.abs(action).max()):.3f} "
        f"foot_dxz_mean={foot_stats['foot_dxz_mean']:.4f} "
        f"foot_dxz_max={foot_stats['foot_dxz_max']:.4f} "
        f"foot_vxz_mean={foot_stats['foot_vxz_mean']:.4f} "
        f"foot_vxz_max={foot_stats['foot_vxz_max']:.4f}"
    )


def main() -> None:
    args = parse_args()
    env = Go1Env(
        render_mode="human",
        max_steps=args.max_steps,
        terrain=args.terrain,
        enable_motors=not args.disable_motors,
        friction_range=(args.friction_min, args.friction_max),
        reset_noise_level=args.reset_noise_level,
        reset_noise_components=args.reset_noise_components,
        spawn_x=args.spawn_x,
        spawn_z=args.spawn_z,
        ground_height_offset=args.ground_height_offset,
    )
    _, reset_info = env.reset()
    tracked_feet = foot_bodies(env)
    reset_foot_xz = foot_xz_positions(tracked_feet)

    print(
        f"viewing zero-action env terrain={args.terrain} "
        f"friction=({args.friction_min:.6g}, {args.friction_max:.6g}) "
        f"motors={'off' if args.disable_motors else 'on'} "
        f"reset_noise={args.reset_noise_level}/{args.reset_noise_components} "
        f"spawn=({args.spawn_x:.2f}, {args.spawn_z:.2f}) "
        f"ground_height_offset={args.ground_height_offset:.2f}"
    )
    print(f"reset_noise_sample={reset_info['reset_noise']}")
    print(f"tracking feet={', '.join(FOOT_BODY_NAMES)}")
    step = 0
    episode = 1

    try:
        while env.render():
            action = np.zeros(12, dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)

            if args.log_interval > 0 and step % args.log_interval == 0:
                foot_stats = foot_debug_stats(tracked_feet, reset_foot_xz)
                print_step(episode, step, action, info, foot_stats)

            ended = truncated or (terminated and not args.ignore_termination)
            if ended:
                print(
                    f"ep={episode:03d} ended "
                    f"reason={info.get('termination_reason') or 'truncated'} "
                    f"steps={step + 1}"
                )
                if args.no_reset_on_end:
                    break
                _, reset_info = env.reset()
                print(f"reset_noise_sample={reset_info['reset_noise']}")
                tracked_feet = foot_bodies(env)
                reset_foot_xz = foot_xz_positions(tracked_feet)
                step = 0
                episode += 1
            else:
                step += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
