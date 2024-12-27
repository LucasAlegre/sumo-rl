import os
import sys
import pickle
import fire

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

sys.path.append('../')
import mysumo.envs  # 确保自定义环境被注册
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from mysumo.envs.resco_envs import grid4x4, arterial4x4, cologne1, cologne3, cologne8, ingolstadt1, ingolstadt7, ingolstadt21


def run(use_gui=False, episodes=1, load_model=False, save_model=True):
    fixed_tl = False
    env = arterial4x4(out_csv_name="outputs/grid4x4/arterial4x4", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    env.reset()

    if load_model and os.path.exists('models/multi_agent_model.pkl'):
        with open('models/multi_agent_model.pkl', 'rb') as f:
            agents = pickle.load(f)
    else:
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

    if save_model:
        with open('models/multi_agent_model.pkl', 'wb') as f:
            pickle.dump(agents, f)

    env.close()


def predict(use_gui=True, episodes=1):
    fixed_tl = False
    env = arterial4x4(out_csv_name="outputs/grid4x4/arterial4x4_predict", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    env.reset()

    if os.path.exists('models/multi_agent_model.pkl'):
        with open('models/multi_agent_model.pkl', 'rb') as f:
            agents = pickle.load(f)
    else:
        print("No saved model found. Please train the model first.")
        return

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = {agent: False for agent in env.agents}

        while env.agents:
            actions = {ts_id: agents[ts_id].act(obs[ts_id]) for ts_id in obs.keys()}
            next_obs, r, terminated, truncated, _ = env.step(actions=actions)
            obs = next_obs

    env.close()


if __name__ == "__main__":
    fire.Fire({
        'run': run,
        'predict': predict
    })

"""
程序有错，不再解决。
"""