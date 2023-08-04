import math, numpy as np
import argparse
import os
import sys
import pandas as pd
import pickle

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == '__main__':

    alpha = 0.005
    gamma = 0.95
    decay = 0.99
    runs = 10

    # env = SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
    #                     route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
    #                     use_gui=False,
    #                     num_seconds=80000,
    #                     min_green=5,
    #                     delta_time=5,
    #                     sumo_seed = 0)
    
    # initial_states = env.reset()
    
    # ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
    #                              state_space=env.observation_space,
    #                              action_space=env.action_space,
    #                              alpha=alpha,
    #                              gamma=gamma,
    #                              exploration_strategy=EpsilonGreedy(initial_epsilon=1.0, min_epsilon=1e-6, decay=decay)) for ts in env.ts_ids}

    # with open('outputs/grid4x4/ql_first_step.pkl','wb') as f:
    #     pickle.dump(ql_agents, f)

    for run in range(1, runs+1):
        env = SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                        route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                        use_gui=False,
                        num_seconds=80000,
                        min_green=5,
                        delta_time=5,
                        sumo_seed = int(run-1))
    
        initial_states = env.reset()
        
        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                    state_space=env.observation_space,
                                    action_space=env.action_space,
                                    alpha=alpha,
                                    gamma=gamma,
                                    exploration_strategy=EpsilonGreedy(initial_epsilon=1.0, min_epsilon=1e-6, decay=decay)) for ts in env.ts_ids}
        if run != 1:
            initial_states = env.reset()
            for ts in initial_states.keys():
                ql_agents[ts].state = env.encode(initial_states[ts], ts)

        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            # actions = {ts: math.floor(np.random.randint(0, 2)) for ts in ql_agents.keys()}
            # print(actions)
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
            s, r, done, info = env.step(action=actions)

            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
            print('time step:', info['step_time'])
            print('info:', info)
            print('/-------------------------------------/')
            # for agent_id in s.keys():
            #     print(f'info: {ql_agents[agent_id].__dict__}\n')
            #     print(agent_id)
            # break
        with open(f"outputs/grid4x4/ql_last_step_{run}.pkl",'wb') as f:
            pickle.dump(ql_agents, f)
        env.save_csv('outputs/grid4x4/ql_train', run)
        env.close()


