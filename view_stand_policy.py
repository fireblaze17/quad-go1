"""View a trained Chrono Go1 standing policy in Irrlicht."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from go1_env import Go1Env, _HOME_JOINT_ANGLES, _JOINT_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy",
        type=Path,
        nargs="?",
        default=Path("runs/stand/final_model.zip"),
        help="Path to a Stable-Baselines3 policy zip.",
    )
    parser.add_argument("--terrain", choices=["flat", "scm"], default="flat")
    parser.add_argument("--friction-min", type=float, default=0.8)
    parser.add_argument("--friction-max", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=1000)
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
    return parser.parse_args()


def print_joint_debug(obs) -> None:
    joint_pos = obs[13:25]
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


def print_policy_step(episode: int, step: int, action, info: dict) -> None:
    terms = info.get("reward_terms", {})
    print(
        f"ep={episode:03d} step={step:04d} "
        f"height={_term(terms, 'trunk_y'):.3f} "
        f"upright={_term(terms, 'upright_score'):.3f} "
        f"axis=({_term(terms, 'trunk_x_up'):+.3f},"
        f"{_term(terms, 'trunk_y_up'):+.3f},"
        f"{_term(terms, 'trunk_z_up'):+.3f}) "
        f"ang_xyz=({_term(terms, 'ang_vel_x'):+.3f},"
        f"{_term(terms, 'ang_vel_y'):+.3f},"
        f"{_term(terms, 'ang_vel_z'):+.3f}) "
        f"xz_vel=({_term(terms, 'lin_vel_x'):+.3f},"
        f"{_term(terms, 'lin_vel_z'):+.3f}) "
        f"act_mean={_term(terms, 'mean_abs_action'):.3f} "
        f"act_max={float(abs(action).max()):.3f} "
        f"dact_mean={_term(terms, 'mean_abs_action_delta'):.3f} "
        f"dact_max={_term(terms, 'max_abs_action_delta'):.3f} "
        f"tilt={_term(terms, 'tilt_error'):.4f} "
        f"pose_err={_term(terms, 'pose_error'):.4f} "
        f"jvel_mean={_term(terms, 'mean_abs_joint_vel'):.3f} "
        f"jvel_max={_term(terms, 'max_abs_joint_vel'):.3f} "
        f"pen=(ctrl:{_term(terms, 'control_penalty'):.4f},"
        f"tilt:{_term(terms, 'tilt_penalty'):.4f},"
        f"pose:{_term(terms, 'pose_penalty'):.4f},"
        f"jvel:{_term(terms, 'joint_vel_penalty'):.4f},"
        f"rate:{_term(terms, 'action_rate_penalty'):.4f},"
        f"ang:{_term(terms, 'ang_vel_penalty'):.4f},"
        f"xz:{_term(terms, 'xz_vel_penalty'):.4f})"
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
    )
    model = PPO.load(args.policy)
    obs, _ = env.reset()
    if args.joint_debug:
        print_joint_debug(obs)
    print(
        f"viewing policy={args.policy} terrain={args.terrain} "
        f"friction=({args.friction_min:.2f}, {args.friction_max:.2f})"
    )
    step = 0
    episode = 1

    try:
        while env.render():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if args.log_interval > 0 and step % args.log_interval == 0:
                print_policy_step(episode, step, action, info)
            step += 1
            if terminated or truncated:
                print(
                    f"ep={episode:03d} ended "
                    f"reason={info.get('termination_reason') or 'truncated'} "
                    f"steps={step}"
                )
                obs, _ = env.reset()
                if args.joint_debug:
                    print_joint_debug(obs)
                step = 0
                episode += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
