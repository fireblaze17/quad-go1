import time

import mujoco.viewer
from stable_baselines3 import PPO

from go1_env import Go1Env


env = Go1Env()
model = PPO.load("go1_stand_policy_v2")

obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        viewer.sync()
        time.sleep(env.model.opt.timestep)

        if terminated or truncated:
            obs, info = env.reset()
