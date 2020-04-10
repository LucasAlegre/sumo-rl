import gym

import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
from gym import spaces
import numpy as np
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.util.gen_route import write_route_file
import traci

from stable_baselines import DQN

env = SumoEnvironment(net_file='nets/big-intersection/big-intersection.net.xml',
                        single_agent=True,
                        route_file='nets/big-intersection/routes.rou.xml',
                        out_csv_name='outputs/big-intersection/dqn',
                        use_gui=False,
                        num_seconds=5400,
                        yellow_time=4,
                        min_green=5,
                        max_green=60,
                        max_depart_delay=0,
                        time_to_load_vehicles=0,
                        phases=[
                        traci.trafficlight.Phase(30, "GGGGrrrrrrGGGGrrrrrr"),  
                        traci.trafficlight.Phase(4, "yyyyrrrrrryyyyrrrrrr"),
                        traci.trafficlight.Phase(15, "rrrrGrrrrrrrrrGrrrrr"),   
                        traci.trafficlight.Phase(4, "rrrryrrrrrrrrryrrrrr"),
                        traci.trafficlight.Phase(30, "rrrrrGGGGrrrrrrGGGGr"),   
                        traci.trafficlight.Phase(4, "rrrrryyyyrrrrrryyyyr"),
                        traci.trafficlight.Phase(15, "rrrrrrrrrGrrrrrrrrrG"), 
                        traci.trafficlight.Phase(4, "rrrrrrrrryrrrrrrrrry")])

model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    buffer_size=50000,
    exploration_fraction=0.05,
    exploration_final_eps=0.02
)
model.learn(total_timesteps=100000)
