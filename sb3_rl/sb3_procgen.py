from procgen import ProcgenEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor

# ProcgenEnv is already vectorized
venv = ProcgenEnv(num_envs=2, env_name="starpilot")

# To use only part of the observation:
# venv = VecExtractDictObs(venv, "rgb")

# Wrap with a VecMonitor to collect stats and avoid errors
venv = VecMonitor(venv=venv)

model = PPO("MultiInputPolicy", venv, verbose=1)
model.learn(10_000)

#  此程序不能运行，因为没有安装procgen。该组件长期没有维护，与新版环境不兼容。
