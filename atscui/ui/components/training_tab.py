import os
import shlex
from pathlib import Path

import gradio as gr
from typing import Generator, Tuple

from stable_baselines3.common.evaluation import evaluate_policy

from atscui.config import TrainingConfig
from atscui.environment.env_creator import createEnv
from atscui.models.agent_creator import createAgent
from atscui.utils.utils import write_eval_result, write_predict_result, extract_crossname_from_netfile, make_sub_dir, write_loop_state


def parseParams(net_file,  # 网络模型
                rou_file,  # 交通需求
                algo_name="DQN",  # 算法名称
                operation="TRAIN",  # 操作名称
                tensorboard_logs="logs",  # tensorboard_logs folder
                single_agent=True,  # 单智能体
                num_seconds=10000,  # 每回合episode仿真步(时长)
                n_eval_episodes=10,  # 评估回合数
                n_steps=1024,  # A2C价值网络更新间隔时间步
                total_timesteps=864_000,  # 总训练时间步（1天)
                gui=True,  # 图形界面
                render_mode=None,  # 渲染模式
                ):
    _cross_name = extract_crossname_from_netfile(net_file)
    cvs_file = _cross_name + "-" + algo_name
    csv_path = os.path.join(make_sub_dir("outs"), cvs_file)
    model_file = _cross_name + "-model-" + algo_name + ".zip"
    model_path = os.path.join(make_sub_dir("models"), model_file)
    predict_file = _cross_name + "-predict-" + algo_name + ".json"
    predict_path = os.path.join(make_sub_dir("predicts"), predict_file)
    eval_file = _cross_name + "-eval-" + algo_name + ".txt"
    eval_path = os.path.join(make_sub_dir("evals"), eval_file)
    tensorboard_logpath = make_sub_dir(tensorboard_logs)

    training_config = TrainingConfig(
        net_file=net_file,
        rou_file=rou_file,
        csv_path=csv_path,
        model_path=model_path,
        predict_path=predict_path,
        eval_path=eval_path,
        single_agent=single_agent,
        gui=gui,
        render_mode=render_mode,
        operation=operation,
        algo_name=algo_name,
        total_timesteps=total_timesteps,
        num_seconds=num_seconds,
        n_steps=n_steps,
        n_eval_episodes=n_eval_episodes,
        tensorboard_logs=tensorboard_logpath)

    print("training_config: {}".format(training_config))
    return training_config


class TrainingTab:
    def __init__(self):
        self.network_file = None
        self.demand_file = None
        self.progress = None

    def render(self):
        with gr.Row():
            with gr.Column(scale=2):
                network_file = gr.File(label="路网模型", value="zszx/net/zszx-2.net.xml", file_types=[".xml", ".net.xml"])
                demand_file = gr.File(label="交通需求", value="zszx/net/zszx-perhour-3.rou.xml", file_types=[".xml", ".rou.xml"])
            with gr.Column(scale=1):
                algorithm = gr.Dropdown(["DQN", "PPO", "A2C", "SAC"], value="PPO", label="算法模型")
                operation = gr.Dropdown(["EVAL", "TRAIN", "PREDICT", "ALL"], value="TRAIN", label="运行功能")

        with gr.Row():
            total_timesteps = gr.Slider(1000, 100_000_000, value=100_000_000, step=1000, label="训练步数")
            num_seconds = gr.Slider(1000, 20_000, value=20_000, step=1000, label="仿真秒数")
        gui_checkbox = gr.Checkbox(label="GUI", value=False)

        run_button = gr.Button("开始运行", variant="primary")
        progress = gr.Slider(minimum=0, maximum=total_timesteps.value, value=0, label="进度", interactive=False)
        output_msg = gr.Textbox(label="输出信息", lines=5)

        run_button.click(
            self.run_training,
            inputs=[network_file, demand_file, algorithm, operation, total_timesteps, num_seconds, gui_checkbox],
            outputs=[progress, output_msg]
        )

    def run_training(self,
                     network_file,
                     demand_file,
                     algorithm,
                     operation,
                     total_timesteps,
                     num_seconds,
                     gui_checkbox) -> Generator[Tuple[int, str], None, None]:

        if not network_file or not demand_file:
            yield 0, "请上传路网模型和交通需求文件"

        if not isinstance(total_timesteps, int) or total_timesteps <= 0 or not isinstance(num_seconds, int) or num_seconds <= 0:
            yield 0, "训练步数和仿真数必须是正整数"

        network_path = shlex.quote(network_file.name)
        demand_path = shlex.quote(demand_file.name)

        use_gui = True if gui_checkbox else False

        training_config = parseParams(network_path, demand_path,
                                      algorithm, operation,
                                      tensorboard_logs="logs",
                                      total_timesteps=total_timesteps,
                                      num_seconds=num_seconds,
                                      gui=use_gui)

        for progress, output_msg in self.run_simulation(training_config):
            yield progress, output_msg

    def run_simulation(self, config):
        env = createEnv(config)
        model = createAgent(env, config).model
        model_obj = Path(config.model_path)
        if model_obj.exists():
            print("load model=====加载训练模型==在原来基础上训练")
            model.load(model_obj)
        progress = 0
        output = ""
        if config.operation == "EVAL":
            output += "evaluate policy====训练前，评估模型\n"
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=config.n_eval_episodes)
            write_eval_result(mean_reward, std_reward, config.eval_path)
            output += f"Mean reward: {mean_reward}, Std reward: {std_reward}\n"
            progress = 100
        elif config.operation == "TRAIN":
            output += "train model=====训练模型，总时间步，进度条\n"
            print(output)
            yield 1, output
            # model.learn(total_timesteps=total_timesteps, progress_bar=True)  # 训练总时间步，100000
            # for i in range(0, config.total_timesteps, 1000):  # 每1000步更新一次进度
            model.learn(total_timesteps=config.total_timesteps, progress_bar=True)
            # progress = int((i + 1000) / config.total_timesteps * 100)
            # yield progress, output
            # print("save model=====保存训练模型")
            output += "save model=====保存训练模型\n"
            print(output)
            yield 2, output
            model.save(config.model_path)
            output += "evaluate policy====训练后，评估模型"
            print(output)
            yield 3, output
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=config.n_eval_episodes)
            write_eval_result(mean_reward, std_reward, config.eval_path)
            output += f"Mean reward: {mean_reward}, Std reward: {std_reward}\n"
            print(output)
            yield progress, output
        elif config.operation == "PREDICT":
            # print("predict====使用模型进行预测")
            output += "predict====使用模型进行预测\n"
            env = model.get_env()
            obs = env.reset()
            info_list = []
            state_list = []
            for i in range(100):
                action, state = model.predict(obs)
                obs, reward, dones, info = env.step(action)
                info_list.append(info[0])
                state_list.append(f"{obs}, {action}, {reward}\n")
                env.render()
                progress = int((i + 1) / 10 * 100)
                yield progress, output
            write_predict_result(info_list, filename=config.predict_path)
            write_loop_state(state_list, filename=config.predict_path)
            output += "Prediction completed and saved.\n"
            print(f"{output}")
            yield progress, output
        elif config.operation == "ALL":
            print("evaluate policy====训练前，评估模型")
            yield 0, output
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=config.n_eval_episodes)
            write_eval_result(mean_reward, std_reward, config.eval_path)
            output += f"Mean reward: {mean_reward}, Std reward: {std_reward}\n"
            print(f"train model=====训练模型，总时间步，进度条:{output}")
            yield 1, output
            model.learn(total_timesteps=config.total_timesteps, progress_bar=True)  # 训练总时间步
            print("save model=====保存训练模型")
            model.save(config.model_path)
            # 评测模型
            model.load(config.model_path)
            print("evaluate policy====训练后，评估模型")
            yield 2, output
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=config.n_eval_episodes)
            write_eval_result(mean_reward, std_reward, config.eval_path)
            output += f"Mean reward: {mean_reward}, Std reward: {std_reward}\n"
            print(f"predict====使用模型进行预测:{output}")
            env = model.get_env()
            obs = env.reset()
            info_list = []
            for i in range(100):
                action, state = model.predict(obs)
                obs, reward, dones, info = env.step(action)
                info_list.append(info[0])
                env.render()
                progress = int((i + 1) / 10 * 100)
                yield progress, f"进度: {progress}%\n正在执行{config.operation}操作..."
            write_predict_result(info_list, filename=config.predict_path)
            output += "Prediction completed and saved.\n"
            print(f"predict====模型预测结束:{output}")
            yield 100, output
        env.close()
        del model

        yield 100, f"{config.operation}操作完成！"
