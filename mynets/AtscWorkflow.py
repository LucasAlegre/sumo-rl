import datetime
import ntpath
import os
import sys
import argparse
from pathlib import Path

from stable_baselines3 import PPO, A2C, SAC

sys.path.append('..')

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.dqn import DQN
import mysumo.envs  # 确保自定义环境被注册
import json
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from mysumo.envs.sumo_env import SumoEnv, ContinuousSumoEnv


def extract_crossname(path):
    # 使用 ntpath.basename 来处理 Windows 路径
    filename = ntpath.basename(path)
    # 分割文件名和扩展名
    name_parts = filename.split('.')
    # 返回第一个部分（基本文件名）
    return name_parts[0]


def create_file_if_not_exists(filename):
    # 获取文件所在的目录路径
    directory = os.path.dirname(filename)
    # 如果目录不存在，创建目录
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    # 如果文件不存在，创建文件
    if not os.path.exists(filename):
        try:
            with open(filename, 'w') as f:
                pass  # 创建一个空文件
            print(f"Created file: {filename}")
        except IOError as e:
            print(f"Error creating file {filename}: {e}")
            return False
    else:
        print(f"File already exists: {filename}")
    return True


def add_directory_if_missing(path, directory="./"):
    # 规范化路径分隔符
    path = os.path.normpath(path)
    # 分割路径
    path_parts = os.path.split(path)
    # 检查是否已经包含目录
    if path_parts[0]:
        return path
    else:
        # 如果没有目录，添加指定的目录
        return os.path.join(directory, path_parts[1])


def write_eval_result(mean, std, filename="eval_result.txt"):
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 将时间和变量组合成一行
    line = f"{current_time}, {mean}, {std}\n"

    create_file_if_not_exists(filename)
    # 以写入模式打开文件并写入
    with open(filename, "a") as file:
        file.write(line)
    print(f"Data written to {filename}")


def write_predict_result(data, filename='predict_results.json', print_to_console=False):
    create_file_if_not_exists(filename)

    if print_to_console:
        print(data)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


# 工作流程：1，设计路口网络模型；2，结合网络模型设计交通需求模型；3，编写sumo配置文件；4，运行本程序。
# netconvert --node-files=my-intersection.nod.xml \
#            --edge-files=my-intersection.edg.xml \
#            --connection-files=my-intersection.con.xml \
#            --tllogic-files=my-intersection.tll.xml \
#            --output-file=my-intersection-2.net.xml \
#            --ignore-errors
# run command: EVAL, TRAIN, PREDICT, ALL
# python AtscWorkflow.py -n my-intersection.net.xml -r my-intersection-probability.rou.xml -q SAC -f EVAL -s 5000 -e 10 -l 10000 -t 1024
# python AtscWorkflow.py -n my-intersection.net.xml -r my-intersection-perhour.rou.xml -q SAC -f TRAIN -s 5000 -e 10 -l 10000 -t 1024
# python AtscWorkflow.py -n my-intersection.net.xml -r my-intersection-period.rou.xml -q SAC -f PREDICT -s 5000 -e 10 -l 10000 -t 1024
# python AtscWorkflow.py -n my-intersection.net.xml -r my-intersection-number.rou.xml -q SAC -f ALL -s 20000 -e 20 -l 100000 -t 2024
def parserArgs(args):
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process some integers.")
    # 添加命令行参数
    parser.add_argument('-n', '--net_file', required=True, type=str, help='网络配置文件,net/cross.net.xml')
    parser.add_argument('-r', '--rou_file', required=True, type=str, help='需求配置文件,net/cross.rou.xml')
    parser.add_argument('-o', '--out_csv_name', required=False, type=str, default="out/cross-algo", help='训练过程输出')
    parser.add_argument('-d', '--model_file', required=False, type=str, default="model/cross-model", help='模型保存文件')
    parser.add_argument('-b', '--tensorboard_log', required=False, type=str, default="logs/cross-log", help='tensorboard目录')
    parser.add_argument('-s', '--num_seconds', required=False, type=int, default=20000, help='仿真秒')
    parser.add_argument('-e', '--n_eval_episodes', required=False, type=int, default=10, help='评估回合数')
    parser.add_argument('-l', '--total_timesteps', required=False, type=int, default=100000, help='总训练时间步')
    parser.add_argument('-t', '--n_steps', required=False, type=int, default=2048, help='A2C价值网络更新间隔时间步')
    parser.add_argument('-g', '--gui', required=False, type=bool, default=False)
    parser.add_argument('-a', '--single_agent', required=False, type=bool, default=True)
    parser.add_argument('-m', '--render_mode', required=False, type=str, default=None)
    parser.add_argument('-q', '--algo_name', required=False, type=str, default="DQN", help='算法')
    parser.add_argument('-f', '--func', required=False, type=str, default="ALL", help='功能')

    # 解析命令行参数
    parsed_args = parser.parse_args(args)

    parsed_args.net_file = add_directory_if_missing(parsed_args.net_file, "./net")
    parsed_args.rou_file = add_directory_if_missing(parsed_args.rou_file, "./net")
    cross_name = extract_crossname(parsed_args.net_file)
    parsed_args.out_csv_name = add_directory_if_missing(cross_name, "./out")
    model_file = cross_name + "-model-" + parsed_args.algo_name + ".zip"
    parsed_args.model_file = add_directory_if_missing(model_file, "./model")
    predict_file = cross_name + "-predict-" + parsed_args.algo_name + ".json"
    parsed_args.predict_file = add_directory_if_missing(predict_file, "./predict")
    eval_file = cross_name + "-eval-" + parsed_args.algo_name + ".txt"
    parsed_args.eval_file = add_directory_if_missing(eval_file, "./eval")

    parsed_args.tensorboard_log = add_directory_if_missing(parsed_args.tensorboard_log, "./logs")

    # 使用解析后的参数
    print(f"=====params passed in args =====")
    print(f"net_file, {parsed_args.net_file}")
    print(f"rou_file, {parsed_args.rou_file}.")
    print(f"out_csv_name, {parsed_args.out_csv_name}.")
    print(f"model_file, {parsed_args.model_file}.")
    print(f"predict_file, {parsed_args.predict_file}.")
    print(f"eval_file, {parsed_args.eval_file}.")
    print(f"num_seconds, {parsed_args.num_seconds}.")
    print(f"n_eval_episodes, {parsed_args.n_eval_episodes}.")
    print(f"total_timesteps, {parsed_args.total_timesteps}.")
    print(f"n_steps, {parsed_args.n_steps}.")
    print(f"gui, {parsed_args.gui}.")
    print(f"single_agent, {parsed_args.single_agent}.")
    print(f"render_mode, {parsed_args.render_mode}.")
    print(f"algo_name, {parsed_args.algo_name}.")
    print(f"func, {parsed_args.func}.")

    return parsed_args


if __name__ == "__main__":
    params = parserArgs(sys.argv[1:])

    # sys.exit(0)
    print("=====create env=====")
    env = SumoEnv(
        net_file=params.net_file,
        route_file=params.rou_file,
        out_csv_name=params.out_csv_name + "-" + params.algo_name,
        single_agent=params.single_agent,
        use_gui=params.gui,
        num_seconds=params.num_seconds,  # 仿真秒，最大值20000
        render_mode=params.render_mode,  # 'rgb_array':This system has no OpenGL support.
    )

    # env = RecordEpisodeStatistics(env)
    # video_recorder = RecordVideo(env, video_folder='recording', name_prefix="sumo-env-dqn")
    # 异常：last video not closed? 录制视频不成功。
    print("=====wrap env======")
    env = Monitor(env, "monitor/SumoEnv-v0")
    env = DummyVecEnv([lambda: env])

    # 创建算法模型实例，DQN, 试用PPO,A2C, SAC等替换
    print("=====create Algorythm Model=====")
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
        print("=====create ContinuousEnv for SAC=====")
        env = ContinuousSumoEnv(
            net_file=params.net_file,
            route_file=params.rou_file,
            out_csv_name=params.out_csv_name + "-" + params.algo_name,
            single_agent=params.single_agent,
            use_gui=params.gui,
            num_seconds=params.num_seconds,  # 仿真秒，最大值20000
            render_mode=params.render_mode,  # 'rgb_array':This system has no OpenGL support.
        )
        print("=====env:action_space:", env.action_space)
        env = Monitor(env, "monitor/ContinuousSumoEnv-v0")
        env = DummyVecEnv([lambda: env])

        print("=====create SAC algorythm=====")
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

    model_path = Path(params.model_file)
    if model_path.exists():
        print("load model=====加载训练模型==在原来基础上训练")
        model.load(model_path)

    if params.func == "EVAL":
        print("evaluate policy====训练前，评测模型的收敛指标")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=params.n_eval_episodes)
        write_eval_result(mean_reward, std_reward, params.eval_file)
    elif params.func == "TRAIN":
        print("train model=====训练模型，总时间步，进度条")
        model.learn(total_timesteps=params.total_timesteps, progress_bar=True)  # 训练总时间步，100000
        print("save model=====保存训练模型")
        model.save(params.model_file)
        print("evaluate policy====训练后，评测模型的收敛指标")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=params.n_eval_episodes)
        write_eval_result(mean_reward, std_reward, params.eval_file)
    elif params.func == "PREDICT":
        print("predict====使用模型进行预测")
        env = model.get_env()
        obs = env.reset()
        info_list = []
        for i in range(10):
            action, state = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            info_list.append(info[0])
            env.render()
        write_predict_result(info_list, filename=params.predict_file)
    elif params.func == "ALL":
        print("evaluate policy====训练前，评测模型的收敛指标")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=params.n_eval_episodes)
        write_eval_result(mean_reward, std_reward, params.eval_file)
        print("train model=====训练模型，总时间步，进度条")
        model.learn(total_timesteps=params.total_timesteps, progress_bar=True)  # 训练总时间步，100000
        print("save model=====保存训练模型")
        model.save(params.model_file)
        # 评测模型
        model.load(params.model_file)
        print("evaluate policy====训练后，评测模型的收敛指标")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=params.n_eval_episodes)
        write_eval_result(mean_reward, std_reward, params.eval_file)
        print("predict====使用模型进行预测")
        env = model.get_env()
        obs = env.reset()
        info_list = []
        for i in range(10):
            action, state = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            info_list.append(info[0])
            env.render()
        write_predict_result(info_list, filename=params.predict_file)

    env.close()
    del model

"""
可以正常运行的训练工作流程序。
1，创建网络: .net.xml
2，创建需求: .rou.xml
3，传进参数运行: python AtscWorkflow.py -n my-intersection.net.xml -r my-intersection-number.rou.xml -q SAC -f ALL -s 20000 -e 20 -l 100000 -t 2024
4，检查结果: model-file，eval-result, predict-result
"""