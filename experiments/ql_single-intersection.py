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

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.8, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mg", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=20000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    args = prs.parse_args()

    env = SumoEnvironment(conf_file='nets/single-intersection/single-intersection.sumocfg',
                          use_gui=args.gui,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          custom_phases=[
                            traci.trafficlight.Phase(10000, 10000, 10000, "GGrr"),   # north-south
                            traci.trafficlight.Phase(2000, 2000, 2000, "yyrr"),
                            traci.trafficlight.Phase(20000, 20000, 20000, "rrGG"),   # west-east
                            traci.trafficlight.Phase(2000, 2000, 2000, "rryy")
                            ])

    initial_states = env.reset()
    ql_agents = {ts: QLAgent(starting_state=initial_states[ts],
                             state_space=env.observation_space,
                             action_space=env.action_space,
                             alpha=args.alpha,
                             gamma=args.gamma,
                             exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}

    infos = []
    done = False
    if args.fixed:
        while not done:
            _, _, done, info = env.step({})
            infos.append(info)
    else:
        while not done:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(actions=actions)

            if args.v:
                print('s=', env.radix_decode(ql_agents['t'].state), 'a=', actions['t'], 's\'=', env.radix_decode(s['t']), 'r=', r['t'])

            infos.append(info)

            for agent_id in ql_agents.keys():
                ql_agents[agent_id].learn(new_state=s[agent_id], reward=r[agent_id])
    env.close()

    df = pd.DataFrame(infos)
    df.to_csv('outputs/single-intersection.csv', index=False)

