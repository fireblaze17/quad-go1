"""
view_env.py — Run the Chrono Go1 Gymnasium env with the Irrlicht viewer.

Set terrain="flat" for a fast rigid ground, or terrain="scm" for deformable soil.
"""

import numpy as np
from go1_env import Go1Env

TERRAIN = "flat"  # "flat" or "scm"

env = Go1Env(render_mode="human", max_steps=1000, terrain=TERRAIN)

while env.render():
    action = np.zeros(12, dtype=np.float32)  # zero torque — watch passive fall
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        env.reset()
