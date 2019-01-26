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

    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 30

    env = SumoEnvironment('nets/4x4-Lucas/4x4.sumocfg',
                          use_gui=False,
                          num_seconds=20000,
                          time_to_load_vehicles=300,
                          max_depart_delay=0,
                          phases=[
                            traci.trafficlight.Phase(35000, 35000, 35000, "GGGrrr"),   # north-south
                            traci.trafficlight.Phase(2000, 2000, 2000, "yyyrrr"),
                            traci.trafficlight.Phase(35000, 35000, 35000, "rrrGGG"),   # west-east
                            traci.trafficlight.Phase(2000, 2000, 2000, "rrryyy")
                            ])

    for run in range(runs+1):
        initial_states = env.reset()
        ql_agents = {ts: QLAgent(starting_state=initial_states[ts],
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)) for ts in env.ts_ids}
        infos = []
        done = False
        while not done:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(actions=actions)
            infos.append(info)

            for agent_id in ql_agents.keys():
                ql_agents[agent_id].learn(new_state=s[agent_id], reward=r[agent_id])

        env.close()

        df = pd.DataFrame(infos)
        df.to_csv('testets.csv', index=False)
        #df.to_csv('outputs/artigoc1c2c1c2badstate_alpha{}_gamma{}_decay{}_run{}.csv'.format(alpha, gamma, decay, run), index=False)

