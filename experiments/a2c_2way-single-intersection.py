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
import traci

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

# multiprocess environment
n_cpu = 2   
env = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                    route_file='nets/2way-single-intersection/single-intersection-gen.rou.xml',
                                    out_csv_name='outputs/2way-single-intersection/a2c-contexts-sp',
                                    single_agent=True,
                                    use_gui=True,
                                    num_seconds=100000,
                                    time_to_load_vehicles=120,
                                    max_depart_delay=0,
                                    phases=[
                                        traci.trafficlight.Phase(32000, 32000, 32000, "GGrrrrGGrrrr"),  
                                        traci.trafficlight.Phase(2000, 2000, 2000, "yyrrrryyrrrr"),
                                        traci.trafficlight.Phase(32000, 32000, 32000, "rrGrrrrrGrrr"),   
                                        traci.trafficlight.Phase(2000, 2000, 2000, "rryrrrrryrrr"),
                                        traci.trafficlight.Phase(32000, 32000, 32000, "rrrGGrrrrGGr"),   
                                        traci.trafficlight.Phase(2000, 2000, 2000, "rrryyrrrryyr"),
                                        traci.trafficlight.Phase(32000, 32000, 32000, "rrrrrGrrrrrG"), 
                                        traci.trafficlight.Phase(2000, 2000, 2000, "rrrrryrrrrry")
                                        ]) for i in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1, lr_schedule='constant')
model.learn(total_timesteps=200000)

""" del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render() """