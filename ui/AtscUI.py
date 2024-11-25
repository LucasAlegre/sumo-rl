import ntpath

import gradio as gr
import shlex
import os
import sys
from pathlib import Path
from plot_figures import plot_process, plot_predict, plot_evaluation
from stable_baselines3 import PPO, A2C, SAC
import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.utils import (add_directory_if_missing, extract_crossname_from_netfile,
                      write_eval_result, write_predict_result, get_relative_path,
                      extract_crossname_from_evalfile, get_gradio_file_info)

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

from mysumo.envs.sumo_env import SumoEnv, ContinuousSumoEnv


def make_sub_dir(name) -> str:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    name_dir = os.path.join(file_dir, name)
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)
    if isinstance(name_dir, bytes):
        name_dir = name_dir.decode('utf-8')  # 解码为字符串
    return name_dir


def parseParams(net_file,  # 网络模型
                rou_file,  # 交通需求
                algo_name="DQN",  # 算法名称
                operation="TRAIN",  # 操作名称
                tensorboard_logs="logs",  # tensorboard_logs folder
                single_agent=True,  # 单智能体
                num_seconds=10000,  # 仿真时长
                n_eval_episodes=10,  # 评估回合数
                n_steps=1024,  # A2C价值网络更新间隔时间步
                total_timesteps=100_000,  # 训练时间步
                gui=False,  # 图形界面
                render_mode=None,  # 渲染模式
                ):
    algo_name = algo_name
    net_path = net_file
    rou_path = rou_file
    _cross_name = extract_crossname_from_netfile(net_path)

    cvs_file = _cross_name + "-" + algo_name
    csv_path = os.path.join(make_sub_dir("outs"), cvs_file)

    model_file = _cross_name + "-model-" + algo_name + ".zip"
    model_path = os.path.join(make_sub_dir("models"), model_file)

    predict_file = _cross_name + "-predict-" + algo_name + ".json"
    predict_path = os.path.join(make_sub_dir("predicts"), predict_file)

    eval_file = _cross_name + "-eval-" + algo_name + ".txt"
    eval_path = os.path.join(make_sub_dir("evals"), eval_file)

    tensorboard_logpath = add_directory_if_missing(tensorboard_logs, "./logs")
    single_agent = single_agent
    operation = operation
    num_seconds = num_seconds
    total_timesteps = total_timesteps
    n_eval_episodes = n_eval_episodes
    n_steps = n_steps
    gui = gui
    render_mode = render_mode

    print("model_file: ", model_file)
    print("predict_file: ", predict_file)
    print("eval_file: ", eval_file)
    print("csv_file:", cvs_file)

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


def plot_training_process(file):
    if file is None:
        return "请选择训练过程文件"
    folder_name, filename = get_gradio_file_info(file)
    output_path = plot_process(file.name, folder_name, filename)
    return output_path, f"训练过程图已生成：{output_path}"


def plot_prediction_result(file):
    if file is None:
        return "请选择预测结果文件"
    folder_name, filename = get_gradio_file_info(file)
    output_path = plot_predict(file.name, folder_name, filename)
    return output_path, f"预测结果图已生成：{output_path}"


def plot_eval_result(file):
    if file is None:
        return "请选择评估结果文件"
    folder_name, filename = get_gradio_file_info(file)
    print("=====folder_name=====", folder_name)
    print("=====filename=====", filename)
    eval_filename = ntpath.basename(filename)
    cross_name = extract_crossname_from_evalfile(eval_filename)  # 提取路口名称
    print("=====cross_name=====", cross_name)
    output_path = plot_evaluation(folder_name, cross_name)
    return output_path, f"预测结果图已生成：{output_path}"


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

    env = createEnv(net_path, rou_path, csv_path, num_seconds, isSAC=algo_name == "SAC")
    model = createAgent(algo_name, env, tensorboard_logpath, model_path)

    model_obj = Path(model_path)
    if model_obj.exists():
        print("load model=====加载训练模型==在原来基础上训练")
        model.load(model_obj)

    progress = 0
    output = ""

    if operation == "EVAL":
        output += "evaluate policy====训练前，评估模型\n"
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        write_eval_result(mean_reward, std_reward, eval_path)
        output += f"Mean reward: {mean_reward}, Std reward: {std_reward}\n"
        progress = 100
    elif operation == "TRAIN":
        # print("train model=====训练模型，总时间步，进度条")
        output += "train model=====训练模型，总时间步，进度条\n"
        print("output:", output)
        # model.learn(total_timesteps=total_timesteps, progress_bar=True)  # 训练总时间步，100000
        for i in range(0, total_timesteps, 1000):  # 每1000步更新一次进度
            model.learn(total_timesteps=1000, progress_bar=False)
            progress = int((i + 1000) / total_timesteps * 100)
            yield progress, output
        # print("save model=====保存训练模型")
        output += "save model=====保存训练模型\n"
        print("output:", output)
        model.save(model_path)
        output += "evaluate policy====训练后，评估模型"
        print("output:", output)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        write_eval_result(mean_reward, std_reward, eval_path)
        output += f"Mean reward: {mean_reward}, Std reward: {std_reward}\n"
        print("output:", output)
        yield progress, output
    elif operation == "PREDICT":
        # print("predict====使用模型进行预测")
        output += "predict====使用模型进行预测\n"
        env = model.get_env()
        obs = env.reset()
        info_list = []
        for i in range(10):
            action, state = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            info_list.append(info[0])
            env.render()
            progress = int((i + 1) / 10 * 100)
            yield progress, output
        write_predict_result(info_list, filename=predict_path)
        output += "Prediction completed and saved.\n"
    elif operation == "ALL":
        # print("evaluate policy====训练前，评估模型")
        output += "evaluate policy====训练前，评估模型\n"
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        write_eval_result(mean_reward, std_reward, eval_path)
        output += f"Mean reward: {mean_reward}, Std reward: {std_reward}\n"
        # print("train model=====训练模型，总时间步，进度条")
        output += "train model=====训练模型，总时间步，进度条\n"
        model.learn(total_timesteps=total_timesteps, progress_bar=True)  # 训练总时间步，100000
        # print("save model=====保存训练模型")
        output += "save model=====保存训练模型\n"
        model.save(model_path)
        # 评测模型
        model.load(model_path)
        # print("evaluate policy====训练后，评估模型")
        output += "evaluate policy====训练后，评估模型\n"
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        write_eval_result(mean_reward, std_reward, eval_path)
        output += f"Mean reward: {mean_reward}, Std reward: {std_reward}\n"
        # print("predict====使用模型进行预测")
        output += "predict====使用模型进行预测\n"
        env = model.get_env()
        obs = env.reset()
        info_list = []
        for i in range(10):
            action, state = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            info_list.append(info[0])
            env.render()
            progress = int((i + 1) / 10 * 100)
            yield progress, f"进度: {progress}%\n正在执行{operation}操作..."
        write_predict_result(info_list, filename=predict_path)
        output += "Prediction completed and saved.\n"

    env.close()
    del model

    yield 100, f"{operation}操作完成！"


def run_button_click(network_file, demand_file, algorithm, operation, total_timesteps, num_seconds):
    for progress, output in run_simulation(network_file, demand_file, algorithm, operation, total_timesteps, num_seconds):
        yield progress, output


def validate_file(file):
    if file is None:
        return False
    _, ext = os.path.splitext(file.name)
    return ext.lower() in ['.xml', '.net.xml', '.rou.xml']  # 根据实际需求调整允许的文件类型


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 交通信号智能体训练系统")

    with gr.Tabs():
        with gr.TabItem("模型训练"):
            with gr.Row():
                with gr.Column(scale=2):
                    network_file = gr.File(label="路网模型", value="mynets/net/my-intersection.net.xml", file_types=[".xml", ".net.xml"])
                    demand_file = gr.File(label="交通需求", value="mynets/net/my-intersection-perhour.rou.xml", file_types=[".xml", ".rou.xml"])
                with gr.Column(scale=1):
                    algorithm = gr.Dropdown(["DQN", "PPO", "A2C", "SAC"], value="DQN", label="算法模型")
                    operation = gr.Dropdown(["EVAL", "TRAIN", "PREDICT", "ALL"], value="TRAIN", label="运行功能")

            with gr.Row():
                total_timesteps = gr.Slider(1000, 100000, value=100000, step=1000, label="训练步数")
                num_seconds = gr.Slider(1000, 20000, value=20000, step=1000, label="仿真秒数")

            run_button = gr.Button("开始运行", variant="primary")
            progress = gr.Slider(0, 100, value=0, label="进度", interactive=False)
            output = gr.Textbox(label="输出信息", lines=5)

        with gr.TabItem("结果可视化"):
            with gr.Row():
                with gr.Column(scale=2):
                    train_process_file = gr.File(label="选择训练过程文件", file_types=[".csv"])
                    plot_train_button = gr.Button("绘制训练过程图", variant="secondary")

            with gr.Row():
                with gr.Column(scale=2):
                    predict_result_file = gr.File(label="选择预测结果文件", file_types=[".json"])
                    plot_predict_button = gr.Button("绘制预测结果图", variant="secondary")

            with gr.Row():
                with gr.Column(scale=2):
                    eval_result_file = gr.File(label="选择评估文件", file_types=[".txt"])
                    plot_eval_button = gr.Button("绘制评估结果图", variant="secondary")

            plot_output = gr.Textbox(label="绘图输出", lines=2)
            plot_image = gr.Image(label="生成的图形")

    run_button.click(
        run_button_click,
        inputs=[network_file, demand_file, algorithm, operation, total_timesteps, num_seconds],
        outputs=[progress, output]
    )

    plot_train_button.click(
        plot_training_process,
        inputs=[train_process_file],
        outputs=[plot_image, plot_output])
    plot_predict_button.click(
        plot_prediction_result,
        inputs=[predict_result_file],
        outputs=[plot_image, plot_output])
    plot_eval_button.click(
        plot_eval_result,
        inputs=[eval_result_file],
        outputs=[plot_image, plot_output])

demo.launch()

"""
模型训练，评估，预测，结果分析。
适用算法：DQN,PPO,A2C,SAC。

运行正常，结果正确。

改进：
1，为配置参数设计一个结构，分训练参数、预测参数、评估参数、log参数等，为不同的算法设计不同的配置项；
2，包及路径的规范化设计。
3，模型的部署与使用。

"""