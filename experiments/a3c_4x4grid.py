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
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.tune.registry import register_env
from gym import spaces
import numpy as np
from sumo_rl.environment.env import SumoEnvironment
import traci


if __name__ == '__main__':
    ray.init()

    register_env("4x4grid", lambda _: SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                    route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                    out_csv_name='outputs/4x4grid/a3c-4x4grid',
                                                    use_gui=False,
                                                    num_seconds=80000,
                                                    time_to_load_vehicles=120,
                                                    max_depart_delay=0,
                                                    phases=[
                                                        traci.trafficlight.Phase(35, "GGGrrr"),   # north-south
                                                        traci.trafficlight.Phase(2, "yyyrrr"),
                                                        traci.trafficlight.Phase(35, "rrrGGG"),   # west-east
                                                        traci.trafficlight.Phase(2, "rrryyy")
                                                        ]))

    trainer = A3CTrainer(env="4x4grid", config={
        "multiagent": {
            "policy_graphs": {
                '0': (A3CTFPolicy, spaces.Box(low=np.zeros(11), high=np.ones(11)), spaces.Discrete(2), {})
            },
            "policy_mapping_fn": lambda id: '0'  # Traffic lights are always controlled by this policy
        },
        "lr": 0.0001,
    })
    while True:
        print(trainer.train())  # distributed training step