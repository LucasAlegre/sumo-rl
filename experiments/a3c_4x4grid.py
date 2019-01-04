import argparse
import os
import sys
import pandas as pd
import ray
from ray.rllib.agents.a3c.a3c import A3CAgent
from ray.rllib.agents.a3c.a3c_tf_policy_graph import A3CPolicyGraph
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
                                          use_gui=True,
                                          num_seconds=20000,
                                          time_to_load_vehicles=300,
                                          max_depart_delay=0,
                                          custom_phases=[
                                            traci.trafficlight.Phase(42000, 42000, 42000, "GGGrrr"),   # north-south
                                            traci.trafficlight.Phase(2000, 2000, 2000, "yyyrrr"),
                                            traci.trafficlight.Phase(42000, 42000, 42000, "rrrGGG"),   # west-east
                                            traci.trafficlight.Phase(2000, 2000, 2000, "rrryyy")
                                            ]))

    trainer = A3CAgent(env="4x4grid", config={
        "multiagent": {
            "policy_graphs": {
                str(id): (A3CPolicyGraph, SumoEnvironment.observation_space, SumoEnvironment.action_space, {}) for id in range(16)
            },
            "policy_mapping_fn": lambda id: id  # Traffic lights are always controlled by this policy
        },
        "lr": 0.0005,
    })
    while True:
        print(trainer.train())  # distributed training step