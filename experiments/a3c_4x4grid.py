import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
import ray
from ray.rllib.agents.a3c.a3c import A3CAgent
from ray.rllib.agents.a3c.a3c_tf_policy_graph import A3CPolicyGraph
from ray.tune.registry import register_env
from gym import spaces
import numpy as np
from environment.env import SumoEnvironment
import traci


if __name__ == '__main__':
    ray.init()

    register_env("4x4grid", lambda _: SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                    route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                    use_gui=False,
                                                    num_seconds=80000,
                                                    time_to_load_vehicles=120,
                                                    max_depart_delay=0,
                                                    phases=[
                                                        traci.trafficlight.Phase(35000, 35000, 35000, "GGGrrr"),   # north-south
                                                        traci.trafficlight.Phase(2000, 2000, 2000, "yyyrrr"),
                                                        traci.trafficlight.Phase(35000, 35000, 35000, "rrrGGG"),   # west-east
                                                        traci.trafficlight.Phase(2000, 2000, 2000, "rrryyy")
                                                        ]))

    trainer = A3CAgent(env="4x4grid", config={
        "multiagent": {
            "policy_graphs": {
                '0': (A3CPolicyGraph, spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1])), spaces.Discrete(2), {})
            },
            "policy_mapping_fn": lambda id: '0'  # Traffic lights are always controlled by this policy
        },
        "lr": 0.0001,
    })
    while True:
        print(trainer.train())  # distributed training step