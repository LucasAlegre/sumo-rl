from stable_baselines3 import PPO
from stable_baselines3.common.envs import SimpleMultiObsEnv

# Stable Baselines provides SimpleMultiObsEnv as an example environment with Dict observations
env = SimpleMultiObsEnv(random_start=False)

# model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000)

# model.save("./model/ppo_multi_input")

# del model

model = PPO.load("./model/ppo_multi_input", env=env)
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, info = env.step(action)
    env.render("human")
