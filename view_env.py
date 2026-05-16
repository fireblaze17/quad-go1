"""Run the Chrono Go1 Gymnasium environment with the Irrlicht viewer."""

import numpy as np

from diagnostics import FOOT_BODY_NAMES, foot_bodies, foot_debug_stats, foot_xz_positions
from go1_env import Go1Env


TERRAIN = "flat"  # Use "scm" for deformable soil.
ENABLE_MOTORS = True
MAX_STEPS = 1000
LOG_INTERVAL = 100


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
        f"height={float(terms.get('trunk_y', 0.0)):.3f} "
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
    env = Go1Env(
        render_mode="human",
        max_steps=MAX_STEPS,
        terrain=TERRAIN,
        enable_motors=ENABLE_MOTORS,
    )
    env.reset()
    tracked_feet = foot_bodies(env)
    reset_foot_xz = foot_xz_positions(tracked_feet)

    print(
        f"viewing zero-action env terrain={TERRAIN} "
        f"motors={'on' if ENABLE_MOTORS else 'off'}"
    )
    print(f"tracking feet={', '.join(FOOT_BODY_NAMES)}")
    step = 0
    episode = 1

    try:
        while env.render():
            action = np.zeros(12, dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)

            if step % LOG_INTERVAL == 0:
                foot_stats = foot_debug_stats(tracked_feet, reset_foot_xz)
                print_step(episode, step, action, info, foot_stats)

            if terminated or truncated:
                print(
                    f"ep={episode:03d} ended "
                    f"reason={info.get('termination_reason') or 'truncated'} "
                    f"steps={step + 1}"
                )
                env.reset()
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
