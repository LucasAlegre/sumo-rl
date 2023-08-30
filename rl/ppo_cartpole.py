from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("./model/ppo_cartpole")

del model  # remove to demonstrate saving and loading

model = PPO.load("./model/ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    img = env.render("rgb_array")
    plt.imshow(img)
    if dones.any():
        obs = env.reset()

# 不能生成图片。
