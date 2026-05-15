"""Run the Chrono Go1 Gymnasium environment with the Irrlicht viewer."""

import numpy as np

from go1_env import Go1Env


TERRAIN = "flat"  # Use "scm" for deformable soil.
ENABLE_MOTORS = True
MAX_STEPS = 1000
LOG_INTERVAL = 100


def print_step(episode: int, step: int, action: np.ndarray, info: dict) -> None:
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
        f"amax={float(np.abs(action).max()):.3f}"
    )


def main() -> None:
    env = Go1Env(
        render_mode="human",
        max_steps=MAX_STEPS,
        terrain=TERRAIN,
        enable_motors=ENABLE_MOTORS,
    )

    print(
        f"viewing zero-action env terrain={TERRAIN} "
        f"motors={'on' if ENABLE_MOTORS else 'off'}"
    )
    step = 0
    episode = 1

    try:
        while env.render():
            action = np.zeros(12, dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)

            if step % LOG_INTERVAL == 0:
                print_step(episode, step, action, info)

            if terminated or truncated:
                print(
                    f"ep={episode:03d} ended "
                    f"reason={info.get('termination_reason') or 'truncated'} "
                    f"steps={step + 1}"
                )
                env.reset()
                step = 0
                episode += 1
            else:
                step += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
