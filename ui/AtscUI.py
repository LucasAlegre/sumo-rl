import gradio as gr
import time
import subprocess
import shlex
import datetime
import ntpath
import os
import sys
import argparse
from pathlib import Path

from stable_baselines3 import PPO, A2C, SAC

sys.path.append('..')

from ui.utils import add_directory_if_missing, extract_crossname, write_eval_result, write_predict_result

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.dqn import DQN
import mysumo.envs  # 确保自定义环境被注册
import json
import os

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from mysumo.envs.sumo_env import SumoEnv, ContinuousSumoEnv


def parseParams(net_file,  # 网络模型
                rou_file,  # 交通需求
                algo_name="DQN",  # 算法名称
                operation="ALL",  # 操作名称
                tensorboard_logs="logs",  # tensorboard_logs folder
                single_agent=True,  # 单智能体
                num_seconds=20000,  # 仿真时长
                n_eval_episodes=20,  # 评估回合数
                n_steps=1024,  # A2C价值网络更新间隔时间步
                total_timesteps=1000000,  # 训练时间步
                gui=False,  # 图形界面
                render_mode=None,  # 渲染模式
                ):
    net_path = add_directory_if_missing(net_file, "./net")
    rou_path = add_directory_if_missing(rou_file, "./net")
    _cross_name = extract_crossname(net_path)
    csv_path = add_directory_if_missing(_cross_name, "./out")
    algo_name = algo_name
    model_file = _cross_name + "-model-" + algo_name + ".zip"
    model_path = add_directory_if_missing(model_file, "./model")
    predict_file = _cross_name + "-predict-" + algo_name + ".json"
    predict_path = add_directory_if_missing(predict_file, "./predict")
    eval_file = _cross_name + "-eval-" + algo_name + ".txt"
    eval_path = add_directory_if_missing(eval_file, "./eval")
    tensorboard_logpath = add_directory_if_missing(tensorboard_logs, "./logs")
    single_agent = single_agent
    operation = operation
    num_seconds = num_seconds
    total_timesteps = total_timesteps
    n_eval_episodes = n_eval_episodes
    n_steps = n_steps
    gui = gui
    render_mode = render_mode

    print("==========AtscUI-parseParams-net_path={}".format(net_path))
    print("==========AtscUI-parseParams-rou_path={}".format(rou_path))
    print("==========AtscUI-parseParams-_cross_name={}".format(_cross_name))
    print("==========AtscUI-parseParams-csv_path={}".format(csv_path))
    print("==========AtscUI-parseParams-algo_name={}".format(algo_name))
    print("==========AtscUI-parseParams-model_path={}".format(model_path))
    print("==========AtscUI-parseParams-predict_path={}".format(predict_path))
    print("==========AtscUI-parseParams-eval_path={}".format(eval_path))
    print("==========AtscUI-parseParams-tensorboard_logpath={}".format(tensorboard_logpath))
    print("==========AtscUI-parseParams-single_agent={}".format(single_agent))
    print("==========AtscUI-parseParams-operation={}".format(operation))
    print("==========AtscUI-parseParams-num_seconds={}".format(num_seconds))
    print("==========AtscUI-parseParams-total_timesteps={}".format(total_timesteps))
    print("==========AtscUI-parseParams-n_eval_episodes={}".format(n_eval_episodes))
    print("==========AtscUI-parseParams-n_steps={}".format(n_steps))
    print("==========AtscUI-parseParams-gui={}".format(gui))
    print("==========AtscUI-parseParams-render_mode={}".format(render_mode))

    return (net_path, rou_path, algo_name, operation,
            csv_path, model_path, predict_path, eval_path, tensorboard_logpath,
            single_agent, num_seconds, n_eval_episodes, n_steps, total_timesteps, gui, render_mode)


def createEnv(net_file, rou_file, csv_name, num_seconds=20000, render_mode=None, single_agent=True, gui=False, isSAC=False):
    print("==========AtscUI-createEnv-csv_name={}".format(csv_name))
    if isSAC:
        print("=====create ContinuousEnv for SAC=====")
        env = ContinuousSumoEnv(
            net_file=net_file,
            route_file=rou_file,
            out_csv_name=csv_name,
            single_agent=single_agent,
            use_gui=gui,
            num_seconds=num_seconds,  # 仿真秒，最大值20000
            render_mode=render_mode,  # 'rgb_array':This system has no OpenGL support.
        )
    else:
        env = SumoEnv(
            net_file=net_file,
            route_file=rou_file,
            out_csv_name=csv_name,
            single_agent=single_agent,
            use_gui=gui,
            num_seconds=num_seconds,  # 仿真秒，最大值20000
            render_mode=render_mode,  # 'rgb_array':This system has no OpenGL support.
        )

    print("=====env:action_space:", env.action_space)
    env = Monitor(env, "monitor/SumoEnv-v0")
    env = DummyVecEnv([lambda: env])

    print("==========AtscUI-createEnv-env.out_csv_name={}".format(env.get_attr("out_csv_name")))

    return env


def createAgent(algo_name, env_name, tensorboard_log, model_file, n_steps=1024, learning_rate=1e-3, gamma=0.9):
    if algo_name == "DQN":
        model = DQN(
            env=env_name,
            policy="MlpPolicy",
            learning_rate=learning_rate,
            learning_starts=0,
            train_freq=1,
            target_update_interval=1000,  # 目标网络更新时间间隔，1000仿真秒
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    elif algo_name == "PPO":
        model = PPO(
            env=env_name,
            policy="MlpPolicy",
            learning_rate=learning_rate,
            n_steps=n_steps,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    elif algo_name == "A2C":
        model = A2C(
            policy='MlpPolicy',
            env=env_name,  # env=make_vec_env(MyWrapper, n_envs=8),  # 使用N个环境同时训练
            learning_rate=learning_rate,
            n_steps=n_steps,  # 运行N步后执行更新,batch_size=n_steps*环境数量
            gamma=gamma,
            tensorboard_log=tensorboard_log,
            verbose=0)
    elif algo_name == "SAC":
        print("=====create SAC algorythm=====")
        model = SAC(
            policy='MlpPolicy',
            env=env_name,  # 使用N个环境同时训练
            learning_rate=learning_rate,
            buffer_size=10_0000,  # reply_buffer_size
            learning_starts=100,  # 积累N步的数据以后开始训练
            batch_size=256,  # 每次采样N条数据
            tau=5e-3,  # target网络软更新系数
            gamma=gamma,
            train_freq=(1, 'step'),  # 训练频率
            tensorboard_log=tensorboard_log,
            verbose=0)
    else:
        raise NotImplementedError

    model_path = Path(model_file)
    if model_path.exists():
        print("load model=====加载训练模型==在原来基础上训练")
        model.load(model_path)

    return model


def run_simulation(network_file, demand_file, algorithm, operation, total_timesteps, num_seconds):
    if not network_file or not demand_file:
        return 0, "请上传路网模型和交通需求文件"

    if not isinstance(total_timesteps, int) or total_timesteps <= 0 or not isinstance(num_seconds, int) or num_seconds <= 0:
        return 0, "训练步数和仿真数必须是正整数"

    network_path = shlex.quote(network_file.name)
    demand_path = shlex.quote(demand_file.name)

    # todo parseParams, createEnv and createAgent
    net_path, rou_path, algo_name, operation, \
        csv_path, model_path, predict_path, eval_path, tensorboard_logpath, \
        single_agent, num_seconds, n_eval_episodes, n_steps, total_timesteps, gui, render_mode \
        = parseParams(network_path, demand_path, algorithm, operation,
                      tensorboard_logs="logs", total_timesteps=total_timesteps, num_seconds=num_seconds)

    env = createEnv(net_path, rou_path, csv_path, num_seconds)
    model = createAgent(algo_name, env, tensorboard_logpath, model_path)

    model_obj = Path(model_path)
    if model_obj.exists():
        print("load model=====加载训练模型==在原来基础上训练")
        model.load(model_obj)

    if operation == "EVAL":
        print("evaluate policy====训练前，评估模型")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        write_eval_result(mean_reward, std_reward, eval_path)
    elif operation == "TRAIN":
        print("train model=====训练模型，总时间步，进度条")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)  # 训练总时间步，100000
        print("save model=====保存训练模型")
        model.save(model_path)
        print("evaluate policy====训练后，评估模型")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        write_eval_result(mean_reward, std_reward, eval_path)
    elif operation == "PREDICT":
        print("predict====使用模型进行预测")
        env = model.get_env()
        obs = env.reset()
        info_list = []
        for i in range(10):
            action, state = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            info_list.append(info[0])
            env.render()
        write_predict_result(info_list, filename=predict_path)
    elif operation == "ALL":
        print("evaluate policy====训练前，评估模型")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        write_eval_result(mean_reward, std_reward, eval_path)
        print("train model=====训练模型，总时间步，进度条")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)  # 训练总时间步，100000
        print("save model=====保存训练模型")
        model.save(model_path)
        # 评测模型
        model.load(model_path)
        print("evaluate policy====训练后，评估模型")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        write_eval_result(mean_reward, std_reward, predict_path)
        print("predict====使用模型进行预测")
        env = model.get_env()
        obs = env.reset()
        info_list = []
        for i in range(10):
            action, state = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            info_list.append(info[0])
            env.render()
        write_predict_result(info_list, filename=predict_path)

    env.close()
    del model

    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # start_time = time.time()
    # progress = 0
    # while process.poll() is None:
    #     elapsed_time = time.time() - start_time
    #     # 假设模拟总时长为100秒，实际应用中应根据SUMO输出动态调整
    #     progress = min(int(elapsed_time), 100)
    #     yield progress, "模拟运行中..."
    #     time.sleep(0.1)
    #
    # stdout, stderr = process.communicate()
    # if process.returncode == 0:
    #     return 100, "模拟完成"
    # else:
    #     return 0, f"模拟出错：{stderr.decode('utf-8')}"


def validate_file(file):
    if file is None:
        return False
    _, ext = os.path.splitext(file.name)
    return ext.lower() in ['.xml', '.net.xml', '.rou.xml']  # 根据实际需求调整允许的文件类型


def view_evaluation():
    return "评估值查看功能已执行。实际应用中，这里应该返回具体的评估结果。"


def view_training_graph():
    return "训练图生成功能已执行。实际应用中，这里应该生成并返回训练过程的图表。"


def view_prediction_graph():
    return "预测图生成功能已执行。实际应用中，这里应该生成并返回预测结果的图表。"


with gr.Blocks() as demo:
    gr.Markdown("# 交通信号智能体训练系统")

    with gr.Row():
        network_file = gr.File(label="路网模型", value="../mynets/net/my-intersection.net.xml", file_types=[".xml", ".net.xml"])
        demand_file = gr.File(label="交通需求", value="../mynets/net/my-intersection-perhour.rou.xml", file_types=[".xml", ".rou.xml"])

    with gr.Row():
        algorithm = gr.Dropdown(["DQN", "PPO", "A2C", "SAC"], value="DQN", label="算法模型")
        operation = gr.Dropdown(["EVAL", "TRAIN", "PREDICT", "ALL"], value="EVAL", label="运行功能")

    with gr.Row():
        total_timesteps = gr.Number(value=1000000, label="训练步数", precision=0)
        num_seconds = gr.Number(value=20000, label="仿真秒数", precision=0)

    run_button = gr.Button("开始运行")
    progress = gr.Slider(0, 100, value=0, label="进度", interactive=False)

    with gr.Row():
        eval_button = gr.Button("查看评估值")
        train_graph_button = gr.Button("查看训练图")
        pred_graph_button = gr.Button("查看预测图")

    output = gr.Textbox(label="输出")

    run_button.click(
        run_simulation,
        inputs=[network_file, demand_file, algorithm, operation, total_timesteps, num_seconds],
        outputs=[progress, output]
    )

    eval_button.click(view_evaluation, outputs=output)
    train_graph_button.click(view_training_graph, outputs=output)
    pred_graph_button.click(view_prediction_graph, outputs=output)

demo.launch()
