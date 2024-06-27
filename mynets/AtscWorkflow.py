import os
import sys
import argparse

from stable_baselines3 import PPO, A2C, SAC

sys.path.append('..')

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.dqn import DQN
import mysumo.envs  # 确保自定义环境被注册

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from mysumo.envs.sumo_env import SumoEnv


# run command: python AtscWorkflow.py -n my-intersection/my-intersection.net.xml -r my-intersection/my-intersection.rou.xml -o out/wf-my-intersection-algo -s 5000 -e 10 -l 10000 -t 1024 -q DQN
def main(args):
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process some integers.")
    # 添加命令行参数
    parser.add_argument('-n', '--net_file', required=False, type=str,
                        default="my-intersection/my-intersection.net.xml",
                        help='net configuration file')
    parser.add_argument('-r', '--rou_file', required=False, type=str,
                        default="my-intersection/my-intersection.rou.xml",
                        help='demand configuration file')
    parser.add_argument('-o', '--out_csv_name', required=False, type=str,
                        default="out/my-intersection-algo")
    parser.add_argument('-d', '--model_file', required=False, type=str,
                        default="model/my-intersection-model")
    parser.add_argument('-b', '--tensorboard_log', required=False, type=str,
                        default="out/tensorboard/wf-my-intersection-log")
    parser.add_argument('-s', '--num_seconds', required=False, type=int, default=20000,
                        help='num seconds (default: 20000)')
    parser.add_argument('-e', '--n_eval_episodes', required=False, type=int, default=10)
    parser.add_argument('-l', '--total_timesteps', required=False, type=int, default=100000)
    parser.add_argument('-t', '--n_steps', required=False, type=int, default=2048)
    parser.add_argument('-g', '--gui', required=False, type=bool, default=False)
    parser.add_argument('-a', '--single_agent', required=False, type=bool, default=True)
    parser.add_argument('-m', '--render_mode', required=False, type=str, default=None)
    parser.add_argument('-q', '--algo_name', required=False, type=str, default="DQN")

    # 解析命令行参数
    parsed_args = parser.parse_args(args)

    # 使用解析后的参数
    print(f"net_file, {parsed_args.net_file}")
    print(f"rou_file, {parsed_args.rou_file}.")
    print(f"out_csv_name, {parsed_args.out_csv_name}.")
    print(f"model_file, {parsed_args.model_file}.")
    print(f"num_seconds, {parsed_args.num_seconds}.")
    print(f"n_eval_episodes, {parsed_args.n_eval_episodes}.")
    print(f"total_timesteps, {parsed_args.total_timesteps}.")
    print(f"n_steps, {parsed_args.n_steps}.")
    print(f"gui, {parsed_args.gui}.")
    print(f"single_agent, {parsed_args.single_agent}.")
    print(f"render_mode, {parsed_args.render_mode}.")
    print(f"algo_name, {parsed_args.algo_name}.")

    return parsed_args


if __name__ == "__main__":
    params = main(sys.argv[1:])
    print(params.net_file)

    # sys.exit(0)

    env = SumoEnv(
        net_file=params.net_file,
        route_file=params.rou_file,
        out_csv_name=params.out_csv_name,
        single_agent=params.single_agent,
        use_gui=params.gui,
        num_seconds=params.num_seconds,  # 仿真秒，最大值20000
        render_mode=params.render_mode,  # 'rgb_array':This system has no OpenGL support.
    )

    # env = RecordEpisodeStatistics(env)
    # video_recorder = RecordVideo(env, video_folder='recording', name_prefix="sumo-env-dqn")
    # 异常：last video not closed? 录制视频不成功。
    env = Monitor(env, "monitor/SumoEnv-v0")
    env = DummyVecEnv([lambda: env])
    model_file = params.model_file + params.algo_name

    # 创建算法模型实例，DQN, 试用PPO,A2C, SAC等替换
    if params.algo_name == "DQN":
        model = DQN(
            env=env,
            policy="MlpPolicy",
            learning_rate=0.001,
            learning_starts=0,
            train_freq=1,
            target_update_interval=1000,  # 目标网络更新时间间隔，1000仿真秒
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
            tensorboard_log=params.tensorboard_log,
            verbose=1,
        )

    elif params.algo_name == "PPO":
        model = PPO(
            env=env,
            policy="MlpPolicy",
            learning_rate=0.001,
            n_steps=2048,
            tensorboard_log=params.tensorboard_log,
            verbose=1,
        )
    elif params.algo_name == "A2C":
        model = A2C(
            policy='MlpPolicy',
            env=env,  # env=make_vec_env(MyWrapper, n_envs=8),  # 使用N个环境同时训练
            learning_rate=1e-3,
            n_steps=5,  # 运行N步后执行更新,batch_size=n_steps*环境数量
            gamma=0.9,
            tensorboard_log=params.tensorboard_log,
            verbose=0)
    elif params.algo_name == "SAC":
        model = SAC(
            policy='MlpPolicy',
            env=env,  # 使用N个环境同时训练
            learning_rate=1e-3,
            buffer_size=10_0000,  # reply_buffer_size
            learning_starts=100,  # 积累N步的数据以后开始训练
            batch_size=256,  # 每次采样N条数据
            tau=5e-3,  # target网络软更新系数
            gamma=0.9,
            train_freq=(1, 'step'),  # 训练频率
            tensorboard_log=params.tensorboard_log,
            verbose=0)
    else:
        raise NotImplementedError

    # 评测模型
    print("evaluate policy====训练前，评测模型的收敛指标")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=params.n_eval_episodes)
    print(mean_reward, std_reward)

    print("load model=====加载训练模型==在原来基础上训练")
    model.load(model_file)
    print("train model=====训练模型，总时间步，进度条")
    model.learn(total_timesteps=params.total_timesteps, progress_bar=True)  # 训练总时间步，100000
    print("save model=====保存训练模型")
    model.save(model_file)
    print("load model=====加载训练模型")
    model.load(model_file)

    # 评测模型
    print("evaluate policy====训练后，评测模型的收敛指标")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=params.n_eval_episodes)
    print(mean_reward, std_reward)

    print("predict====使用模型进行预测")
    env = model.get_env()
    obs = env.reset()
    for i in range(10):
        action, state = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        print("\n======", i, "======")
        print(" obs:", i, obs)
        print(" reward:", reward)
        print(" dones:", dones)
        print(" info:", info)
        env.render()
    env.close()
    del model
