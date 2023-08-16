import os
import gymnasium as gym
# import pybullet_envs

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

# Note: pybullet is not compatible yet with Gymnasium
# you might need to use `import rl_zoo3.gym_patches`
# and use gym (not Gymnasium) to instantiate the env
# Alternatively, you can use the MuJoCo equivalent "HalfCheetah-v4"
# vec_env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
vec_env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
# Automatically normalize the input features and reward
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                       clip_obs=10.)

model = PPO("MlpPolicy", vec_env)
model.learn(total_timesteps=2000)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "./tmp/"
model.save(log_dir + "ppo_halfcheetah")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
vec_env.save(stats_path)

# To demonstrate loading
del model, vec_env

# Load the saved statistics
vec_env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
vec_env = VecNormalize.load(stats_path, vec_env)
#  do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False

# Load the agent
model = PPO.load(log_dir + "ppo_halfcheetah", env=vec_env)
