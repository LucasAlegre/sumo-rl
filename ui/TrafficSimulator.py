import json
import ntpath
import gradio as gr
import shlex
import os
import sys
from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback

from plot_figures import plot_process, plot_predict, plot_evaluation
from stable_baselines3 import PPO, A2C, SAC
import matplotlib.pyplot as plt
import logging
from logging.handlers import RotatingFileHandler

sys.path.append('..')

from ui.utils import (add_directory_if_missing, extract_crossname_from_netfile,
                      write_eval_result, write_predict_result, get_relative_path,
                      extract_crossname_from_evalfile, get_gradio_file_info, create_file_if_not_exists)

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.dqn.dqn import DQN
import mysumo.envs  # 确保自定义环境被注册

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from mysumo.envs.sumo_env import SumoEnv, ContinuousSumoEnv


class TrafficSimulator:
    def __init__(self, algorithm='DQN'):
        self.algorithm = algorithm
        self.env = None
        self.model = None
        self.setup_logger()
        self.config = self.load_config()

    def setup_logger(self):
        self.logger = logging.getLogger('TrafficSimulator')
        self.logger.setLevel(logging.DEBUG)
        log_file = './logs/traffic_simulator.log'
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def load_config(self):
        config_file = f'config/{self.algorithm.lower()}_config.json'
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_file} not found. Using default configuration.")
            return self.get_default_config()

    def save_config(self):
        os.makedirs('./config', exist_ok=True)
        config_file = f'config/{self.algorithm.lower()}_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        self.logger.info(f"Configuration saved to {config_file}")

    def get_default_config(self):
        return {
            'env_params': {
                "net_file": "../mynets/net/my-intersection.net.xml",
                "route_file": "../mynets/net/my-intersection-perhour.rou.xml",
                'delta_time': 5,
                'yellow_time': 2,
                'min_green': 5,
                'max_green': 50,
                'num_seconds': 3600,
                'reward_fn': 'queue',
                "out_csv_name": './outs/process.csv',
                'single_agent': True,
                'use_gui': False
            },
            'algo_params': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                "tensorboard_log": "./tensorboard/",
            },
            'train_params': {
                'total_timesteps': 1000000,
                'eval_freq': 10000,
                'n_eval_episodes': 5,
            },
            'operation': 'evaluation',
            'algorithm': 'PPO',
            'model_path': './models',
            'eval_path': './evals',
            'predict_path': './predicts',
        }

    def update_config(self, **kwargs):
        """Update config with UI inputs"""
        algorithm = kwargs.get('algorithm')
        if algorithm is None:
            algorithm = 'DQN'
        net_file = kwargs.get('net_file')
        _cross_name = "cross-" + algorithm.lower()
        if net_file is not None:
            _cross_name = extract_crossname_from_netfile(net_file)

        for key, value in kwargs.items():
            if key == 'algorithm':
                algorithm = value
            for section in ['env_params', 'algo_params', 'train_params']:
                if key in self.config[section]:
                    self.config[section][key] = value
                    break
            os.makedirs('./outs', exist_ok=True)
            csv_path = './outs/' + _cross_name + "-" + algorithm
            self.config['env_params']['out_csv_name'] = csv_path
            os.makedirs('./models', exist_ok=True)
            self.model_name = _cross_name + "-model-" + algorithm
            self.config['model_path'] = './models'
            predict_path = "./predicts/" + _cross_name + "-predict-" + algorithm + ".json"
            create_file_if_not_exists(predict_path)
            self.config['predict_path'] = predict_path
            eval_path = "./evals/" + _cross_name + "-eval-" + algorithm + ".txt"
            create_file_if_not_exists(eval_path)
            self.config['eval_path'] = eval_path

        self.logger.info(f"Updated config: {self.config}")


    def create_env(self, net_file, rou_file):
        env_params = self.config['env_params']
        env_params['net_file'] = net_file
        env_params['route_file'] = rou_file

        # Use ContinuousSumoEnv for SAC, SumoEnv for others
        if self.algorithm == 'SAC':
            env = ContinuousSumoEnv(**env_params)
        else:
            env = SumoEnv(**env_params)

        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env

    def create_agent(self):
        model_path = Path(self.config['model_path'] + '/' + 'final_' + self.algorithm + '.zip')
        if model_path.exists():
            print("load model=====加载训练模型==在原来基础上训练")

        algo_params = self.config['algo_params']
        if self.algorithm == 'PPO':
            if model_path.exists():
                return PPO.load(path=model_path, env=self.env, **algo_params)
            return PPO('MlpPolicy', self.env, **algo_params)
        elif self.algorithm == 'DQN':
            if model_path.exists():
                return DQN.load(path=model_path, env=self.env, **algo_params)
            return DQN('MlpPolicy', self.env, **algo_params)
        elif self.algorithm == 'A2C':
            if model_path.exists():
                return A2C.load(path=model_path, env=self.env, **algo_params)
            return A2C('MlpPolicy', self.env, **algo_params)
        elif self.algorithm == 'SAC':
            if model_path.exists():
                return SAC.load(path=model_path, env=self.env, **algo_params)
            return SAC('MlpPolicy', self.env, **algo_params)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # 创建并返回智能体模型

    def run_simulation(self, network_file, demand_file, operation):
        self.logger.info(f"Starting {operation} with {self.algorithm} algorithm")
        try:
            self.env = self.create_env(network_file, demand_file)
            self.model = self.create_agent()

            if operation == 'evaluate':
                yield from self.evaluate()
            elif operation == 'train':
                yield from self.train()
            elif operation == 'predict':
                yield from self.predict()
            else:
                raise ValueError("Invalid operation")

        except Exception as e:
            self.logger.exception(f"Error during {operation}: {str(e)}")
            yield 100, f"Error: {str(e)}"
        finally:
            self.close()

    def evaluate(self):
        n_eval_episodes = self.config['train_params']['n_eval_episodes']
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=n_eval_episodes)
        result = f"Evaluation result: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
        self.logger.info(result)
        write_eval_result(mean_reward, std_reward, self.config['eval_path'])
        yield 100, result

    def train(self):
        total_timesteps = self.config['train_params']['total_timesteps']
        eval_freq = self.config['train_params']['eval_freq']
        n_eval_episodes = self.config['train_params']['n_eval_episodes']

        eval_callback = EvalCallback(self.env, best_model_save_path=self.config['model_path'],
                                     log_path=self.config['model_path'], eval_freq=eval_freq,
                                     n_eval_episodes=n_eval_episodes, deterministic=True, render=False)

        for i in range(0, total_timesteps, eval_freq):
            self.model.learn(total_timesteps=eval_freq, callback=eval_callback)
            progress = int((i + eval_freq) / total_timesteps * 100)
            self.logger.debug(f"Training progress: {progress}%")
            yield progress, f"Training progress: {progress}%"

        self.logger.info("Saving trained model")
        self.model.save(os.path.join(self.config['model_path'], f"{self.model_name}"))
        yield 100, "Training completed"

    def predict(self, num_steps=100):
        obs = self.env.reset()
        info_list = []
        for i in range(num_steps):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            info_list.append(info[0])
            progress = int((i + 1) / num_steps * 100)
            yield progress, f"Prediction step {i + 1}/{num_steps}"
            if done:
                obs = self.env.reset()
        write_predict_result(info_list, filename=self.config['predict_path'])
        yield 100, "Prediction completed"

    def load_model(self, path):
        if self.algorithm == 'PPO':
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(path, env=self.env)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(path, env=self.env)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path, env=self.env)

    def close(self):
        self.logger.info("Closing environment and cleaning up resources")
        if self.env:
            self.env.close()
        if self.model:
            del self.model
        self.env = None
        self.model = None
        self.logger.info("Cleanup completed")


# Gradio Interface functions

simulator: TrafficSimulator


def initialize_simulator(algorithm="DQN"):
    global simulator
    simulator = TrafficSimulator(algorithm)
    return f"Initialized {algorithm} simulator"


def update_config(**kwargs):
    if simulator:
        simulator.update_config(**kwargs)
        simulator.save_config()
        return "Configuration updated and saved"
    return "Simulator not initialized"


def run_operation(network_file, demand_file, operation):
    if simulator:
        for progress, output in simulator.run_simulation(network_file, demand_file, operation):
            yield progress, output
    else:
        yield 100, "Simulator not initialized"


def load_model(path):
    if simulator:
        simulator.load_model(path)
        return f"Model loaded from {path}"
    return "Simulator not initialized"


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
    eval_filename = ntpath.basename(filename)
    cross_name = extract_crossname_from_evalfile(eval_filename)
    output_path = plot_evaluation(folder_name, cross_name)
    return output_path, f"评估结果图已生成：{output_path}"
