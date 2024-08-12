import sys
import time
import numpy as np
from typing import Callable, Dict, Any

sys.path.append('..')

from aits.RealWorldEnv import RealWorldEnv
from stable_baselines3 import SAC, PPO, A2C, DQN


class TrafficControlSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = RealWorldEnv(**config.get("env_params"))
        self.algorithm = config.get("algorithm", "DQN")
        self.model_path = config.get("model_path", "")
        self.model = None
        self.load_model(self.model_path)

    def load_model(self, model_path):
        if self.algorithm == 'SAC':
            self.model = SAC.load(model_path)
        elif self.algorithm == 'PPO':
            self.model = PPO.load(model_path)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(model_path)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def get_action(self, observation):
        # 处理整个环境的观察
        if isinstance(observation, dict):
            # 如果观察是字典（每个交叉口一个观察），我们需要将它们连接起来
            obs_array = np.concatenate([obs for obs in observation.values()])
        else:
            # 如果观察已经是一个数组，直接使用它
            obs_array = observation

        # 确保观察的形状正确
        expected_shape = self.model.observation_space.shape
        if obs_array.shape != expected_shape:
            # 如果形状不匹配，进行适当的调整
            if len(obs_array) < expected_shape[0]:
                # 如果观察太短，填充零
                padded_obs = np.zeros(expected_shape)
                padded_obs[:len(obs_array)] = obs_array
                obs_array = padded_obs
            else:
                # 如果观察太长，截断
                obs_array = obs_array[:expected_shape[0]]

        # 使用模型预测动作
        action, _ = self.model.predict(obs_array, deterministic=True)

        return action

    def run(self):
        obs, _ = self.env.reset()  # 是多个路口观察值的字典
        try:
            while True:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)

                print(f"==========action: {action}")
                print(f"Step info: {info}")
                time.sleep(self.env.delta_time)

                if terminated or truncated:
                    obs, _ = self.env.reset()
        except KeyboardInterrupt:
            print("Stopping the system...")
        finally:
            self.env.close()


def main():
    config = {
        'env_params': {
            'intersection_ids': ["intersection_1", "intersection_2"],
            'delta_time': 1,  # 5
            'yellow_time': 1,  # 2
            'min_green': 5,
            'max_green': 30,
            'num_seconds': 360,  # 3600
            'reward_fn': "queue",
            "action_space_type": "discrete",
        },
        'algo_params': {
            'learning_rate': 1e-4,
            'buffer_size': 1000,  # 100000
            'learning_starts': 100,  # 1000
            'batch_size': 64,
            'gamma': 0.99,
            'tau': 0.005,
        },
        'algorithm': 'DQN',
        'total_timesteps': 2000,  # 500000
        "model_path": "models_3/final_DQN_model.zip",
    }

    tcs = TrafficControlSystem(config)
    tcs.run()


if __name__ == "__main__":
    main()
