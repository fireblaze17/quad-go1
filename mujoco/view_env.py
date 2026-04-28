import time 
import mujoco.viewer 
from go1_env import Go1Env

env = Go1Env()
obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()
        time.sleep(env.model.opt.timestep)
        if terminated or truncated:
            obs, info = env.reset()
