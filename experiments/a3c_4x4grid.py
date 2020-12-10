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
from sumo_rl import SumoEnvironment
import traci


if __name__ == '__main__':
    ray.init()

    register_env("4x4grid", lambda _: SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                    route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                    out_csv_name='outputs/4x4grid/a3c',
                                                    use_gui=False,
                                                    num_seconds=80000,
                                                    max_depart_delay=0))

    trainer = A3CTrainer(env="4x4grid", config={
        "multiagent": {
            "policies": {
                '0': (A3CTFPolicy, spaces.Box(low=np.zeros(10), high=np.ones(10)), spaces.Discrete(2), {})
            },
            "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
        },
        "lr": 0.001,
        "no_done_at_end": True
    })
    while True:
        print(trainer.train())  # distributed training step