import argparse
import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from environment.env import SumoEnvironment


if __name__ == '__main__':

    env = SumoEnvironment('nets/4x4-Lucas/4x4.sumocfg', use_gui=False, num_seconds=20000)

    env.reset()
    done = False
    while not done:
        done = env.step({})

    env.close()
