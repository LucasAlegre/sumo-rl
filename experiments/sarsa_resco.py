import os
import sys

import fire


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

from sumo_rl import cologne8


def run(use_gui=False, episodes=50):
    fixed_tl = False

    env = cologne8(out_csv_name="outputs/cologne8/test", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    env.reset()

    agents = {
        ts_id: TrueOnlineSarsaLambda(
            env.observation_spaces[ts_id],
            env.action_spaces[ts_id],
            alpha=0.0001,
            gamma=0.95,
            epsilon=0.05,
            lamb=0.1,
            fourier_order=7,
        )
        for ts_id in env.agents
    }

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = {agent: False for agent in env.agents}

        if fixed_tl:
            while env.agents:
                _, _, terminated, truncated, _ = env.step(None)
        else:
            while env.agents:
                actions = {ts_id: agents[ts_id].act(obs[ts_id]) for ts_id in obs.keys()}

                next_obs, r, terminated, truncated, _ = env.step(actions=actions)

                for ts_id in next_obs.keys():
                    agents[ts_id].learn(
                        state=obs[ts_id],
                        action=actions[ts_id],
                        reward=r[ts_id],
                        next_state=next_obs[ts_id],
                        done=terminated[ts_id],
                    )
                    obs[ts_id] = next_obs[ts_id]

    env.close()


if __name__ == "__main__":
    fire.Fire(run)
