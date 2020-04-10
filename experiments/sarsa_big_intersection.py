import argparse
import os
import sys
import pandas as pd
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.sarsa_lambda import TrueOnlineSarsaLambda


if __name__ == '__main__':

    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/big-intersection/sarsakk' #+ experiment_time

    env = SumoEnvironment(net_file='nets/big-intersection/big-intersection.net.xml',
                          single_agent=True,
                          route_file='nets/big-intersection/routes.rou.xml',
                          out_csv_name=out_csv,
                          use_gui=False,
                          num_seconds=5400,
                          yellow_time=4,
                          min_green=5,
                          max_green=60,
                          max_depart_delay=0,
                          time_to_load_vehicles=0,
                          phases=[
                            traci.trafficlight.Phase(30, "GGGGrrrrrrGGGGrrrrrr"),  
                            traci.trafficlight.Phase(4, "yyyyrrrrrryyyyrrrrrr"),
                            traci.trafficlight.Phase(15, "rrrrGrrrrrrrrrGrrrrr"),   
                            traci.trafficlight.Phase(4, "rrrryrrrrrrrrryrrrrr"),
                            traci.trafficlight.Phase(30, "rrrrrGGGGrrrrrrGGGGr"),   
                            traci.trafficlight.Phase(4, "rrrrryyyyrrrrrryyyyr"),
                            traci.trafficlight.Phase(15, "rrrrrrrrrGrrrrrrrrrG"), 
                            traci.trafficlight.Phase(4, "rrrrrrrrryrrrrrrrrry")])

    fixed_tl = False
    agent = TrueOnlineSarsaLambda(env.observation_space, env.action_space, alpha=0.0001, gamma=0.95, epsilon=0.05, lamb=0.9, fourier_order=21)

    for run in range(1, 10 +1):
        obs = env.reset()
        done = False

        if fixed_tl:
            while not done:
                _, _, done, _ = env.step(None)

        else:
            while not done:
                action = agent.act(agent.get_features(obs))

                next_obs, r, done, _ = env.step(action=action)

                agent.learn(state=obs, action=action, reward=r, next_state=next_obs, done)

                obs = next_obs

        env.save_csv(out_csv, run)




