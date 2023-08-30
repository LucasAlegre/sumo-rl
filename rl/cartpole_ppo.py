from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("./model/cartpole_ppo")

del model  # remove to demonstrate saving and loading

model = PPO.load("./model/cartpole_ppo")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    if dones.any():
        obs = vec_env.reset()

# 运行成功，平衡柱显示正常。
