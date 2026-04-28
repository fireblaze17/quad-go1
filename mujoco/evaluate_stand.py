import numpy as np
from stable_baselines3 import PPO

from go1_env import Go1Env


NUM_EPISODES = 20
POLICY_PATH = "go1_stand_policy_v2"


env = Go1Env()
model = PPO.load(POLICY_PATH)

episode_lengths = []
episode_rewards = []
fall_count = 0

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    if terminated:
        fall_count += 1

    episode_lengths.append(steps)
    episode_rewards.append(total_reward)

    print(
        f"Episode {episode + 1}: "
        f"steps={steps}, "
        f"reward={total_reward:.2f}, "
        f"fell={terminated}"
    )

env.close()

print()
print("Evaluation summary")
print("------------------")
print(f"Episodes: {NUM_EPISODES}")
print(f"Average steps: {np.mean(episode_lengths):.2f}")
print(f"Best steps: {np.max(episode_lengths)}")
print(f"Worst steps: {np.min(episode_lengths)}")
print(f"Average reward: {np.mean(episode_rewards):.2f}")
print(f"Falls: {fall_count}/{NUM_EPISODES}")
print(f"Fall rate: {100.0 * fall_count / NUM_EPISODES:.1f}%")
