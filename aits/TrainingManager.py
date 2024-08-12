import glob
import os
import sys
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
    def __init__(self, verbose=0, log_freq=5000):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'rewards' in info:
            self.rewards.append(info['rewards'])
        if self.n_calls % self.log_freq == 0:
            print("RewardLogger._on_step-info: {}".format(info))
        return True

    def on_training_end(self) -> None:
        # 这里你可以保存或打印累积的奖励信息
        print("Accumulated rewards:", self.rewards)


class PeriodicSaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "model"):
        super(PeriodicSaveCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
        return True


class TrainingManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithm = config.get('algorithm', 'DQN')
        self.action_space_type = self._determine_action_space_type()
        self.env = self._create_env()
        self.model = None
        self.model_path = config.get('model_path', 'models')
        os.makedirs(self.model_path, exist_ok=True)

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
        env = RealWorldEnv(**self.config['env_params'])
        env = FlattenObservationWrapper(env)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env

    def _create_model(self):
        model_cls = {'DQN': DQN, 'PPO': PPO, 'A2C': A2C, 'SAC': SAC}[self.algorithm]
        model_file = os.path.join(self.model_path, f"{self.algorithm}_model.zip")

        if os.path.exists(model_file):
            print(f"Loading existing model from {model_file}")
            self.model = model_cls.load(model_file, env=self.env)
        else:
            print(f"Creating new {self.algorithm} model")
            if self.algorithm == 'DQN':
                if not isinstance(self.env.action_space, spaces.Discrete):
                    raise ValueError("DQN requires a discrete action space.")
            elif self.algorithm == 'SAC':
                if not isinstance(self.env.action_space, spaces.Box):
                    raise ValueError("SAC requires a continuous action space.")

            self.model = model_cls('MlpPolicy', self.env, **self.config.get('algo_params', {}))

        print(f"Action space type: {self.action_space_type}")
        print(f"Action space: {self.env.action_space}")

    def train(self):
        print("=====TrainingManager-train: Start training")
        if self.model is None:
            self._create_model()

        # Create an evaluation callback
        eval_env = self._create_env()
        eval_callback = EvalCallback(eval_env, best_model_save_path=self.model_path,
                                     log_path=self.model_path, eval_freq=2000,
                                     deterministic=True, render=False)
        # 创建定期保存回调
        save_callback = PeriodicSaveCallback(save_freq=1000, save_path=self.model_path,
                                             name_prefix=self.algorithm)
        # Train the model
        reward_logger = RewardLogger(log_freq=1000)
        self.model.learn(total_timesteps=self.config.get('total_timesteps', 10000),
                         callback=[eval_callback, save_callback, reward_logger])

        # Save the final model
        self.model.save(os.path.join(self.model_path, f"final_{self.algorithm}_model"))

    def evaluate(self, num_episodes: int = 10):
        print("=====TrainingManager-evaluate: Start evaluating")
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
            filename = self.get_latest_model_path()
        model_cls = {'DQN': DQN, 'PPO': PPO, 'A2C': A2C, 'SAC': SAC}[self.algorithm]
        self.model = model_cls.load(filename, env=self.env)

    def get_latest_model_path(self):
        model_files = glob.glob(os.path.join(self.model_path, f"{self.algorithm}_*_steps.zip"))
        if not model_files:
            raise FileNotFoundError("No saved models found.")
        latest_model = max(model_files, key=os.path.getctime)
        return latest_model

    def get_best_model_path(self):
        best_model_path = os.path.join(self.model_path, f"best_{self.algorithm}_model.zip")
        if os.path.exists(best_model_path):
            return best_model_path
        else:
            return self.get_latest_model_path()

    def visualize_training(self):
        # This method can be expanded to create more detailed visualizations
        logdir = os.path.join(self.model_path, "logs")
        os.system(f"tensorboard --logdir {logdir}")


def debug_config():
    config = {
        'env_params': {
            'intersection_ids': ["intersection_1", "intersection_2"],  # 使用单个交叉口简化调试
            'delta_time': 1,  # 减少时间步长，加快模拟速度
            'yellow_time': 1,
            'min_green': 2,
            'max_green': 5,
            'num_seconds': 100,  # 减少总模拟时间
            'reward_fn': "queue",
            'action_space_type': 'discrete'  # 明确指定动作空间类型
        },
        'algorithm': 'DQN',  # 使用简单的算法
        'total_timesteps': 200,  # 非常少的训练步数，用于快速运行
        'algo_params': {
            'learning_rate': 0.1,  # 高学习率，快速看到变化
            'buffer_size': 100,  # 小缓冲区，便于追踪数据流
            'learning_starts': 10,  # 很快开始学习
            'batch_size': 4,  # 小批量，便于调试
            'train_freq': 1,  # 每步都训练
            'gradient_steps': 1,
            'exploration_fraction': 0.5,
            'exploration_final_eps': 0.1,
            'verbose': 2,  # 详细输出
            'tensorboard_log': "./debug_logs/",  # 启用TensorBoard日志
            'policy_kwargs': dict(net_arch=[8, 8])  # 小型网络，便于调试
        }
    }

    trainer = TrainingManager(config)
    trainer.train()
    trainer.evaluate()
    trainer.save_model()


def train_config():
    config = {
        'env_params': {
            'intersection_ids': ["intersection_1", "intersection_2", "intersection_3", "intersection_4"],
            'delta_time': 5,
            'yellow_time': 2,
            'min_green': 10,
            'max_green': 60,
            'num_seconds': 3600,  # 每个episode模拟1小时
            'reward_fn': "queue",
            'action_space_type': 'discrete'
        },
        'algorithm': 'PPO',
        'total_timesteps': 2_000_000,
        'algo_params': {
            'policy': 'MultiInputPolicy',  # 更改为MultiInputPolicy
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_sde': False,
            'sde_sample_freq': -1,
            'target_kl': None,
            'tensorboard_log': "./ppo_traffic_tensorboard/",
            'policy_kwargs': dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])],
                activation_fn=nn.ReLU
            ),
            'verbose': 1
        }
    }

    # 创建环境
    env = RealWorldEnv(**config['env_params'])
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 创建模型
    model = PPO(env=env, **config['algo_params'])

    # 创建评估回调
    eval_env = RealWorldEnv(**config['env_params'])
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/',
                                 log_path='./eval_logs/', eval_freq=10000,
                                 deterministic=True, render=False)

    # 训练模型
    model.learn(total_timesteps=config['total_timesteps'], callback=eval_callback)

    # 保存最终模型
    model.save("ppo_traffic_final_model")

    # 训练后评估
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    if sys.argv[1] == 1:
        debug_config()
    else:
        train_config()
