import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from environment.env import SumoEnvironment
from agents.ql_agent import QLAgent
from exploration.epsilon_greedy import EpsilonGreedy


if __name__ == '__main__':

    env = SumoEnvironment('nets/4x4-Lucas/4x4.sumocfg', use_gui=False, num_seconds=20000)

    initial_states = env.reset()
    ql_agents = {ts: QLAgent(starting_state=initial_states[ts],
                             state_space=env.observation_space,
                             action_space=env.action_space,
                             alpha=0.5,
                             gamma=0.95,
                             exploration_strategy=EpsilonGreedy(initial_epsilon=1.0, min_epsilon=0.005, decay=0.998)) for ts in env.ts_ids}

    infos = []
    done = False
    while not done:
        actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

        s, r, done, info = env.step(actions)

        infos.append(info)

        for agent_id in ql_agents.keys():
            ql_agents[agent_id].learn(new_state=s[agent_id], reward=r[agent_id])

    df = pd.DataFrame(infos)
    df.to_csv('outputs/result2.csv')

    env.close()
