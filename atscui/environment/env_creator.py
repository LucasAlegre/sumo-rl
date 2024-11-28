from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mysumo.envs.sumo_env import ContinuousSumoEnv, SumoEnv


def createEnv(config):
    if config.algo_name == "SAC":
        print("=====create ContinuousEnv for SAC=====")
        env = ContinuousSumoEnv(
            net_file=config.net_file,
            route_file=config.rou_file,
            out_csv_name=config.csv_path,
            single_agent=config.single_agent,
            use_gui=config.gui,
            num_seconds=config.num_seconds,  # 仿真秒，最大值20000
            render_mode=config.render_mode,  # 'rgb_array':This system has no OpenGL support.
        )
    else:
        env = SumoEnv(
            net_file=config.net_file,
            route_file=config.rou_file,
            out_csv_name=config.csv_path,
            single_agent=config.single_agent,
            use_gui=config.gui,
            num_seconds=config.num_seconds,  # 仿真秒，最大值20000
            render_mode=config.render_mode,  # 'rgb_array':This system has no OpenGL support.
        )

    print("=====env:action_space:", env.action_space)
    env = Monitor(env, "monitor/SumoEnv-v0")
    env = DummyVecEnv([lambda: env])

    return env
