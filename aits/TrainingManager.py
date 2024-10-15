import argparse
import ast
import os
import sys
import json
from typing import Dict, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn

sys.path.append('..')
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aits.RealWorldEnv import RealWorldEnv


class FlattenObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def _flatten_obs(self, obs):
        return np.concatenate([obs[key].flatten() for key in sorted(obs.keys())])


class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'rewards' in info:
            self.rewards.append(info['rewards'])
        return True

    def on_training_end(self) -> None:
        print("Accumulated rewards:", self.rewards)


class TrainingManager:
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.algorithm = self.config.get('algorithm', 'DQN')
        self.action_space_type = self._determine_action_space_type()
        self.env = self._create_env()
        self.model = None
        self.model_path = self.config.get('model_path', 'models')
        os.makedirs(self.model_path, exist_ok=True)

    def load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, 'r') as file:
            try:
                config = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error parsing config file: {e}")
                raise

        return config

    def _determine_action_space_type(self):
        if self.algorithm in ['SAC']:
            return 'continuous'
        elif self.algorithm in ['DQN', 'PPO', 'A2C']:
            return 'discrete'
        else:
            return 'auto'

    def _create_env(self) -> gym.Env:
        env_params = self.config['env_params']
        env_params['action_space_type'] = self.action_space_type
        env = RealWorldEnv(**env_params)
        env = FlattenObservationWrapper(env)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env

    def _get_activation_fn(self, activation_fn_name: str):
        activation_fns = {
            'ReLU': nn.ReLU,
            'Tanh': nn.Tanh,
            'ELU': nn.ELU,
            'LeakyReLU': nn.LeakyReLU,
            'Sigmoid': nn.Sigmoid
            # Add more activation functions as needed
        }
        return activation_fns.get(activation_fn_name, nn.ReLU)

    def _create_model(self):
        algo_params = self.config.get('algo_params', {})
        if 'policy_kwargs' in algo_params:
            if 'activation_fn' in algo_params['policy_kwargs']:
                activation_fn_name = algo_params['policy_kwargs']['activation_fn']
                algo_params['policy_kwargs']['activation_fn'] = self._get_activation_fn(activation_fn_name)

            # Convert null to None for clip_range_vf and target_kl
            if 'clip_range_vf' in algo_params and algo_params['clip_range_vf'] is None:
                algo_params['clip_range_vf'] = None
            if 'target_kl' in algo_params and algo_params['target_kl'] is None:
                algo_params['target_kl'] = None

        if self.algorithm == 'DQN':
            if not isinstance(self.env.action_space, spaces.Discrete):
                raise ValueError("DQN requires a discrete action space.")
            self.model = DQN('MlpPolicy', self.env, **algo_params)
        elif self.algorithm == 'PPO':
            self.model = PPO('MlpPolicy', self.env, **algo_params)
        elif self.algorithm == 'A2C':
            self.model = A2C('MlpPolicy', self.env, **algo_params)
        elif self.algorithm == 'SAC':
            if not isinstance(self.env.action_space, spaces.Box):
                raise ValueError("SAC requires a continuous action space.")
            self.model = SAC('MlpPolicy', self.env, **algo_params)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        print(f"Action space type: {self.action_space_type}")
        print(f"Action space: {self.env.action_space}")

    def train(self):
        if self.model is None:
            self._create_model()

        eval_env = self._create_env()
        eval_callback = EvalCallback(eval_env, best_model_save_path=self.model_path,
                                     log_path=self.model_path, eval_freq=10000,
                                     deterministic=True, render=False)

        reward_logger = RewardLogger()
        self.model.learn(total_timesteps=self.config.get('total_timesteps', 1000000),
                         callback=[eval_callback, reward_logger])

        self.model.save(os.path.join(self.model_path, f"final_{self.algorithm}_model"))

    def evaluate(self, num_episodes: int = 10):
        if self.model is None:
            self.load_model()

        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=num_episodes)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def save_model(self, filename: str = None):
        if filename is None:
            filename = f"{self.algorithm}_model"
        self.model.save(os.path.join(self.model_path, filename))

    def load_model(self, filename: str = None):
        if filename is None:
            filename = f"{self.algorithm}_model"
        model_cls = {'DQN': DQN, 'PPO': PPO, 'A2C': A2C, 'SAC': SAC}[self.algorithm]
        self.model = model_cls.load(os.path.join(self.model_path, filename), env=self.env)

    def visualize_training(self):
        logdir = os.path.join(self.model_path, "logs")
        os.system(f"tensorboard --logdir {logdir}")


def main():
    parser = argparse.ArgumentParser(description='Run Traffic Control System Training')
    parser.add_argument('operation', choices=['debug', 'test', 'train'], help='Execution parameter')
    parser.add_argument('algorithm', choices=['DQN', 'PPO', 'A2C', 'SAC'], help='Algorithm to use')
    args = parser.parse_args()

    config_file = f"{args.algorithm.lower()}-{args.operation}.json"
    # config_path = os.path.join("config", config_file)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", config_file)

    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)

    trainer = TrainingManager(config_path)
    trainer.train()
    trainer.evaluate()
    trainer.save_model()


if __name__ == "__main__":
    main()

"""
这个 `TrainingManager` 类是一个用于管理强化学习训练过程的综合工具。其功能和逻辑：

1. 初始化 (`__init__`):
   - 加载配置文件
   - 确定使用的算法和动作空间类型
   - 创建环境
   - 设置模型保存路径

2. 配置加载 (`load_config`):
   - 从 JSON 文件加载训练配置

3. 环境创建 (`_create_env`):
   - 创建 `RealWorldEnv` 环境
   - 应用观察空间扁平化包装器
   - 使用 `Monitor` 包装器记录训练数据
   - 创建向量化环境和标准化包装器

4. 模型创建 (`_create_model`):
   - 根据配置的算法（DQN, PPO, A2C, SAC）创建相应的模型
   - 处理模型参数，包括激活函数的设置

5. 训练过程 (`train`):
   - 创建评估回调和奖励记录器
   - 执行模型训练
   - 保存最终模型

6. 评估 (`evaluate`):
   - 对训练好的模型进行评估

7. 模型保存和加载 (`save_model`, `load_model`):
   - 提供保存和加载模型的功能

8. 训练可视化 (`visualize_training`):
   - 使用 TensorBoard 可视化训练日志

9. 主函数 (`main`):
   - 解析命令行参数
   - 根据参数选择配置文件
   - 创建 `TrainingManager` 实例并执行训练、评估和保存过程

主要特点和逻辑：

1. 灵活性：支持多种强化学习算法（DQN, PPO, A2C, SAC）。
2. 可配置性：使用 JSON 配置文件来设置训练参数。
3. 环境适应性：使用包装器来处理观察空间和动作空间的兼容性。
4. 评估和监控：集成了评估回调和奖励记录器。
5. 模型管理：提供了保存和加载模型的功能。
6. 可视化：支持使用 TensorBoard 进行训练过程可视化。

这个程序设计得非常全面，适合用于各种强化学习任务，特别是在交通控制系统的训练中。它提供了从配置加载到模型训练、评估、保存的完整流程，同时保持了良好的模块化和可扩展性。

"""
