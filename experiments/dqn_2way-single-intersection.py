import gym
import numpy as np

from stable_baselines.deepq import DQN, MlpPolicy

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
import traci


if __name__ == '__main__':

    env = SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                    route_file='nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                                    out_csv_name='outputs/2way-single-intersection/dqn-vhvh2-stable-mlp-bs',
                                    single_agent=True,
                                    use_gui=True,
                                    num_seconds=100000,
                                    time_to_load_vehicles=120,
                                    max_depart_delay=0,
                                    phases=[
                                        traci.trafficlight.Phase(32, "GGrrrrGGrrrr"),  
                                        traci.trafficlight.Phase(2, "yyrrrryyrrrr"),
                                        traci.trafficlight.Phase(32, "rrGrrrrrGrrr"),   
                                        traci.trafficlight.Phase(2, "rryrrrrryrrr"),
                                        traci.trafficlight.Phase(32, "rrrGGrrrrGGr"),   
                                        traci.trafficlight.Phase(2, "rrryyrrrryyr"),
                                        traci.trafficlight.Phase(32, "rrrrrGrrrrrG"), 
                                        traci.trafficlight.Phase(2, "rrrrryrrrrry")
                                        ])

    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02
    )
    model.learn(total_timesteps=100000)