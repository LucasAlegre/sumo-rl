import os
import sys

import gymnasium as gym

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file="mynets/my-intersection/my-intersection.net.xml",
    route_file="mynets/my-intersection/my-intersection.rou.xml",
    out_csv_name="mynets/out/big-intersection-dqn",
    single_agent=True,
    use_gui=False,
    num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    learning_starts=0,
    buffer_size=50000,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
    tensorboard_log="./tensorboard/dqn-big-intersection/",
)
model.learn(total_timesteps=100000)

# Save, load, evaluate and predict the model
# model.save("mynets/model/big-intersection-dqn")
# del model

model = DQN.load("mynets/model/big-intersection-dqn", env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

# 在Mac/Ubuntu上，在env:sumoai-sb3-grid4x4中运行成功。
# 1，修改需求，或者说，整理当涂数据，使之成为本试验的需求数据。
# 2，修改算法，尝试使用PPO，QL，A2C, TRPO或者别的算法。
