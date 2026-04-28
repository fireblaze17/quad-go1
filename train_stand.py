from stable_baselines3 import PPO 
from go1_env import Go1Env


env = Go1Env()
model = PPO(
    "MlpPolicy",
    env,
    verbose = 1,
)
model.learn(total_timesteps=100_000)
model.save("go1_stand_policy_v2")
env.close()