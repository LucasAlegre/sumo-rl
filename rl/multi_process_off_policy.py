import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("Pendulum-v1", n_envs=4, seed=0)

# We collect 4 transitions per call to `ènv.step()`
# and performs 2 gradient steps per call to `ènv.step()`
# if gradient_steps=-1, then we would do 4 gradients steps per call to `ènv.step()`
model = SAC("MlpPolicy", vec_env, train_freq=1, gradient_steps=2, verbose=1)
model.learn(total_timesteps=10_000)
model.save("./model/sac_pendulum_v1")

del model  # remove to demonstrate saving and loading

model = SAC.load("./model/sac_pendulum_v1", env=vec_env)
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
