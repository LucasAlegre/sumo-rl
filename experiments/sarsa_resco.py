import os
import sys

import fire

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

from sumo_rl import grid4x4, arterial4x4, cologne1, cologne3, cologne8, ingolstadt1, ingolstadt7, ingolstadt21


def run(use_gui=False, episodes=50):
    fixed_tl = False

    # env = grid4x4(out_csv_name="outputs/grid4x4/grid4x4", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    env = arterial4x4(out_csv_name="outputs/grid4x4/arterial4x4", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = cologne1(out_csv_name="outputs/grid4x4/cologne1", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = cologne3(out_csv_name="outputs/grid4x4/cologne3", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = cologne8(out_csv_name="outputs/grid4x4/cologne8", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = ingolstadt1(out_csv_name="outputs/grid4x4/ingolstadt1", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = ingolstadt7(out_csv_name="outputs/grid4x4/ingolstadt7", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = ingolstadt21(out_csv_name="outputs/grid4x4/ingolstadt21", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)

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
        obs = env.reset()
        done = {agent: False for agent in env.agents}

        if fixed_tl:
            while not done["__all__"]:
                _, _, done, _ = env.step(None)
        else:
            try:
                while not done[env.agents[0]]:
                    for obs_it in obs:
                        actions = {ts_id: agents[ts_id].act(obs_it[ts_id]) for ts_id in obs_it.keys()}
                        next_obs, r, done, truncated, _ = env.step(actions=actions)
                        for ts_id2 in next_obs.keys():
                            agents[ts_id2].learn(
                                state=obs_it[ts_id2], action=actions[ts_id2], reward=r[ts_id2], next_state=next_obs[ts_id2], done=done[ts_id2]
                            )
                            obs_it[ts_id2] = next_obs[ts_id2]
            except:
                print("except=")

    env.close()


if __name__ == "__main__":
    fire.Fire(run)

# 测试说明 2023-10-30
# 1. 检查环境：libs-mac/ubuntu-2023-10-30.txt
# 2. python experiments/sarsa_resco.py
#    ubuntu运行成功，macOS运行正常。

