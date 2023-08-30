# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False

from ray.rllib.algorithms.ppo import PPOConfig

env_name = "CartPole-v1"
env = gym.make(env_name)
algo = PPOConfig().environment(env_name).build()

episode_reward = 0
terminated = truncated = False

if gymnasium:
    obs, info = env.reset()
else:
    obs = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    if gymnasium:
        obs, reward, terminated, truncated, info = env.step(action)
    else:
        obs, reward, terminated, info = env.step(action)
    episode_reward += reward
