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

from sumo_rl import SumoEnvironment


env = SumoEnvironment(
    net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
    single_agent=True,
    route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
    out_csv_name="outputs/big-intersection/dqn",
    use_gui=True,
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
)
model.learn(total_timesteps=100000)
