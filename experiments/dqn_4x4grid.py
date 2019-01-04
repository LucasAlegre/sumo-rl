import argparse
import os
import sys
import pandas as pd
import ray
from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.tune.registry import register_env
from gym import spaces
import numpy as np
from environment.env_ray import SumoEnvironment

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci


if __name__ == '__main__':
    ray.init()

    register_env("4x4grid", lambda _: SumoEnvironment('nets/4x4-Lucas/4x4.sumocfg',
                                          use_gui=False,
                                          num_seconds=20000,
                                          time_to_load_vehicles=300,
                                          max_depart_delay=0,
                                          custom_phases=[
                                            traci.trafficlight.Phase(42000, 42000, 42000, "GGGrrr"),   # north-south
                                            traci.trafficlight.Phase(2000, 2000, 2000, "yyyrrr"),
                                            traci.trafficlight.Phase(42000, 42000, 42000, "rrrGGG"),   # west-east
                                            traci.trafficlight.Phase(2000, 2000, 2000, "rrryyy")
                                            ]))
    '''
    trainer = DQNAgent(env="4x4grid", config={
        "multiagent": {
            "policy_graphs": {
                "ts": (DQNPolicyGraph, spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([1, 50, 1, 1, 1, 1])), spaces.Discrete(2), {}),
            },
            "policy_mapping_fn": lambda _: "ts"  # Traffic lights are always controlled by this policy
        },
        "schedule_max_timesteps": 30000,
        "timesteps_per_iteration": 500,

    })
    while True:
        print(trainer.train())  # distributed training step
    '''
    trainer = DQNAgent(env="4x4grid", config={
        "multiagent": {
            "policy_graphs": {
                str(id): (DQNPolicyGraph, spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1])), spaces.Discrete(2), {}) for id in range(16)
            },
            "policy_mapping_fn": lambda id: id  # Traffic lights are always controlled by this policy
        },
        "schedule_max_timesteps": 30000,
        "timesteps_per_iteration": 100,

    })
    while True:
        print(trainer.train())  # distributed training step