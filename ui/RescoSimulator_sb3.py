import os
import sys
import pickle
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import fire

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sys.path.append('..')
import mysumo.envs  # 确保自定义环境被注册

from mysumo import arterial4x4

def make_env(env_id, rank, seed=0):
    def _init():
        env = arterial4x4(out_csv_name=f"outputs/grid4x4/arterial4x4_{rank}", use_gui=False, yellow_time=2, fixed_ts=False)
        # env.seed(seed + rank)
        return env
    return _init

def run(use_gui=False, episodes=50, load_model=False, save_model=True):
    # 创建环境
    env = DummyVecEnv([make_env("arterial4x4", i) for i in range(1)])

    if load_model and os.path.exists('dqn_resco_model.zip'):
        model = DQN.load("dqn_resco_model")
        model.set_env(env)
    else:
        model = DQN("MlpPolicy", env, verbose=1)

    # 训练模型
    model.learn(total_timesteps=episodes * env.envs[0].agents[0].max_steps)

    if save_model:
        model.save("dqn_resco_model")

    env.close()

def predict(use_gui=True, episodes=1):
    env = arterial4x4(out_csv_name="outputs/grid4x4/arterial4x4_predict", use_gui=use_gui, yellow_time=2, fixed_ts=False)

    if os.path.exists('dqn_resco_model.zip'):
        model = DQN.load("dqn_resco_model")
    else:
        print("No saved model found. Please train the model first.")
        return

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)

    env.close()

if __name__ == "__main__":
    fire.Fire({
        'run': run,
        'predict': predict
    })
    
