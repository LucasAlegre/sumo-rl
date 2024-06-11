import os
import sys
from datetime import datetime

import fire


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

from sumo_rl import SumoEnvironment


def run(use_gui=True, runs=1):
    out_csv = "outputs/double/sarsa-double"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/double/network.net.xml",
        single_agent=False,
        route_file="sumo_rl/nets/double/flow.rou.xml",
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=86400,
        yellow_time=3,
        min_green=5,
        max_green=60,
    )

    fixed_tl = False
    agents = {
        ts_id: TrueOnlineSarsaLambda(
            env.observation_spaces(ts_id),
            env.action_spaces(ts_id),
            alpha=0.000000001,
            gamma=0.95,
            epsilon=0.05,
            lamb=0.1,
            fourier_order=7,
        )
        for ts_id in env.ts_ids
    }

    for run in range(1, runs + 1):
        obs = env.reset()
        done = {"__all__": False}

        if fixed_tl:
            while not done["__all__"]:
                _, _, done, _ = env.step(None)
        else:
            while not done["__all__"]:
                actions = {ts_id: agents[ts_id].act(obs[ts_id]) for ts_id in obs.keys()}

                next_obs, r, done, _ = env.step(action=actions)

                for ts_id in next_obs.keys():
                    agents[ts_id].learn(
                        state=obs[ts_id], action=actions[ts_id], reward=r[ts_id], next_state=next_obs[ts_id], done=done[ts_id]
                    )
                    obs[ts_id] = next_obs[ts_id]

        env.save_csv(out_csv, run)


if __name__ == "__main__":
    fire.Fire(run)
