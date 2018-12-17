import argparse
import os
import sys
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from environment.env import SumoEnvironment
from agents.ql_agent import QLAgent
from exploration.epsilon_greedy import EpsilonGreedy


if __name__ == '__main__':

    verbose = True
    no_learning = False

    env = SumoEnvironment(conf_file='nets/single-intersection/single-intersection.sumocfg',
                          use_gui=True,
                          num_seconds=20000,
                          min_green=10,
                          custom_phases=[
                            traci.trafficlight.Phase(42000, 42000, 42000, "GGrr"),   # north-south
                            traci.trafficlight.Phase(2000, 2000, 2000, "yyrr"),
                            traci.trafficlight.Phase(42000, 42000, 42000, "rrGG"),   # west-east
                            traci.trafficlight.Phase(2000, 2000, 2000, "rryy"),
                            ])

    initial_states = env.reset()
    ql_agents = {ts: QLAgent(starting_state=initial_states[ts],
                             state_space=env.observation_space,
                             action_space=env.action_space,
                             alpha=0.1,
                             gamma=0.8,
                             exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.05, decay=1)) for ts in env.ts_ids}

    infos = []
    done = False

    if no_learning:
        while not done:
            _, _, done, info = env.step({})
            infos.append(info)
    else:
        while not done:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(actions=actions)

            if verbose:
                print('s=', env.radix_decode(ql_agents['t'].state), 'a=', actions['t'], 's\'=', env.radix_decode(s['t']), 'r=', r['t'])

            infos.append(info)

            for agent_id in ql_agents.keys():
                ql_agents[agent_id].learn(new_state=s[agent_id], reward=r[agent_id])
    env.close()

    df = pd.DataFrame(infos)
    df.to_csv('outputs/single-intersection.csv')

