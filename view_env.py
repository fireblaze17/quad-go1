"""Run the Chrono Go1 Gymnasium environment with the Irrlicht viewer."""

import numpy as np

from go1_env import Go1Env


TERRAIN = "flat"  # Use "scm" for deformable soil.
ENABLE_MOTORS = True
MAX_STEPS = 1000


def main() -> None:
    env = Go1Env(
        render_mode="human",
        max_steps=MAX_STEPS,
        terrain=TERRAIN,
        enable_motors=ENABLE_MOTORS,
    )

    try:
        while env.render():
            action = np.zeros(12, dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
