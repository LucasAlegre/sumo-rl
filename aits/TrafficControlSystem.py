import sys
import time
import numpy as np
import gymnasium as gym
from typing import Callable

sys.path.append('..')

from aits.RealWorldEnv import RealWorldEnv
from stable_baselines3 import SAC, PPO, A2C, DQN
from stable_baselines3.common.utils import get_linear_fn


class TrafficControlSystem:
    def __init__(self, env_params, model_paths, algorithm='SAC'):
        self.env = RealWorldEnv(**env_params)
        self.algorithm = algorithm
        self.models = {}
        self.load_models(model_paths)

    def load_models(self, model_paths):
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        for ts in self.env.intersection_ids:
            if self.algorithm == 'SAC':
                self.models[ts] = SAC.load(model_paths[ts], custom_objects=custom_objects)
            elif self.algorithm == 'PPO':
                self.models[ts] = PPO.load(model_paths[ts], custom_objects=custom_objects)
            elif self.algorithm == 'A2C':
                self.models[ts] = A2C.load(model_paths[ts], custom_objects=custom_objects)
            elif self.algorithm == 'DQN':
                self.models[ts] = DQN.load(model_paths[ts], custom_objects=custom_objects)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def get_action(self, ts, observation):
        # 调整观察空间以匹配模型的期望
        adjusted_observation = self.adjust_observation(observation)
        action, _ = self.models[ts].predict(adjusted_observation, deterministic=True)
        if self.algorithm in ['SAC', 'PPO', 'A2C']:  # 连续动作空间
            # 将连续动作映射到离散动作
            discrete_action = int(action * self.env.traffic_signals[ts].num_green_phases)
            return discrete_action
        else:  # DQN，离散动作空间
            return action

    def adjust_observation(self, observation):
        # 如果观察空间的维度不匹配，进行调整
        if len(observation) != 22:
            # 假设我们需要将11维扩展到22维
            # 这里我们简单地重复每个元素一次，您可能需要根据实际情况调整这个逻辑
            adjusted_obs = np.repeat(observation, 2)
            return adjusted_obs
        return observation

    def run(self):
        obs, _ = self.env.reset()
        try:
            while True:
                actions = {}
                for ts in self.env.intersection_ids:
                    traffic_signal = self.env.traffic_signals[ts]
                    if traffic_signal.time_to_act:
                        action = self.get_action(ts, obs[ts])
                        actions[ts] = action

                if isinstance(self.env.action_space, gym.spaces.Dict):
                    obs, reward, terminated, truncated, info = self.env.step(actions)
                else:
                    # 如果动作空间不是字典，我们需要将动作转换为适当的格式
                    if isinstance(self.env.action_space, gym.spaces.Discrete):
                        action = next(iter(actions.values())) if actions else 0  # 如果没有动作，默认为0
                    elif isinstance(self.env.action_space, gym.spaces.Box):
                        action = np.array(list(actions.values()) or [0] * self.env.action_space.shape[0])
                    obs, reward, terminated, truncated, info = self.env.step(action)

                print(f"Step info: {info}")  # 打印每一步的信息
                time.sleep(self.env.delta_time)

                if terminated or truncated:
                    obs, _ = self.env.reset()
        except KeyboardInterrupt:
            print("Stopping the system...")
        finally:
            self.env.close()


def main():
    env_params = {
        "intersection_ids": ["intersection_1", "intersection_2"],
        "delta_time": 5,
        "yellow_time": 2,
        "min_green": 5,
        "max_green": 30,
        "num_seconds": 3600,
        "reward_fn": "queue"
    }

    model_paths = {
        "intersection_1": "./models_2/final_DQN_model.zip",
        "intersection_2": "./models_2/final_DQN_model.zip"
    }

    tcs = TrafficControlSystem(env_params, model_paths, algorithm='DQN')
    tcs.run()


if __name__ == "__main__":
    main()
