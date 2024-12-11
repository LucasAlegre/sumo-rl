import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from typing import Dict, List, Union, Callable

from aits.RealWorldDataCollector import RealWorldDataCollector
from aits.RealWorldTrafficSignal import RealWorldTrafficSignal
from aits.SignalController import SignalController


class RealWorldEnv(gym.Env):
    def __init__(
            self,
            intersection_ids: List[str],
            delta_time: int,
            yellow_time: int,
            min_green: int,
            max_green: int,
            num_seconds: int,
            reward_fn: Union[str, Callable, Dict[str, Union[str, Callable]]],
            action_space_type: str = 'auto'
    ):
        super(RealWorldEnv, self).__init__()

        self.intersection_ids = intersection_ids
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_fn = reward_fn

        self.data_collectors = {
            ts: RealWorldDataCollector(ts) for ts in self.intersection_ids
        }

        self.signal_controllers = {
            ts: SignalController(ts) for ts in self.intersection_ids
        }

        self.traffic_signals = {
            ts: RealWorldTrafficSignal(
                self,
                ts,
                self.delta_time,
                self.yellow_time,
                self.min_green,
                self.max_green,
                0,  # begin_time
                self.reward_fn[ts] if isinstance(self.reward_fn, dict) else self.reward_fn,
                self.data_collectors[ts],
            ) for ts in self.intersection_ids
        }

        self.vehicles = dict()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}

        # 存储每个交叉口的动作数量
        self.actions_per_intersection = {ts: self.traffic_signals[ts].action_space.n for ts in intersection_ids}

        # 创建离散和连续动作空间
        self.discrete_action_space = spaces.Discrete(sum(self.actions_per_intersection.values()))
        self.continuous_action_space = spaces.Box(low=0, high=1, shape=(len(intersection_ids),), dtype=np.float32)

        # 根据 action_space_type 设置实际使用的动作空间
        if action_space_type == 'auto':
            self.action_space = self.discrete_action_space
        elif action_space_type == 'discrete':
            self.action_space = self.discrete_action_space
        elif action_space_type == 'continuous':
            self.action_space = self.continuous_action_space
        else:
            raise ValueError("Invalid action_space_type. Use 'auto', 'discrete', or 'continuous'.")

        # 合并所有交叉口的观察空间
        obs_spaces = {ts: self.traffic_signals[ts].observation_space for ts in self.intersection_ids}
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.start_time = time.time()
        self.next_action_time = self.start_time

        # Reset all traffic signals and data collectors
        for ts in self.intersection_ids:
            self.traffic_signals[ts].signal_controller.set_phase(0)  # Set to initial phase
            self.data_collectors[ts].update()  # Ensure data is fresh

        observations = {ts: self.traffic_signals[ts].compute_observation() for ts in self.intersection_ids}
        return observations, {}  # Return initial observation and an empty info dict

    def step(self, action):
        # Decode the action
        actions = self._decode_action(action)

        # Apply actions
        for ts, act in actions.items():
            if self.traffic_signals[ts].time_to_act:
                self.traffic_signals[ts].set_next_phase(act)

        # Wait for delta_time
        time.sleep(self.delta_time)

        # Update all traffic signals and data collectors
        for ts in self.intersection_ids:
            self.traffic_signals[ts].update()
            self.data_collectors[ts].update()

        # Compute observations
        observations = {ts: self.traffic_signals[ts].compute_observation() for ts in self.intersection_ids}

        # Compute rewards
        rewards = {ts: self.traffic_signals[ts].compute_reward() for ts in self.intersection_ids}
        total_reward = sum(rewards.values())  # 计算总奖励

        # Check if simulation is done
        done = time.time() - self.start_time >= self.num_seconds
        terminated = done
        truncated = False  # We're not truncating episodes in this environment

        # Compute info
        info = self._compute_info()
        info['rewards'] = rewards  # 将单个交叉口的奖励放入 info 字典中

        # print("=====RealWorldEnv-step-info:", info)
        return observations, total_reward, terminated, truncated, info

    def _decode_action(self, action):
        actions = {}
        if isinstance(self.action_space, spaces.Box):
            for i, ts in enumerate(self.intersection_ids):
                discrete_action = int(action[i] * self.actions_per_intersection[ts])
                actions[ts] = discrete_action
        else:
            for ts in self.intersection_ids:
                if action < self.actions_per_intersection[ts]:
                    actions[ts] = action
                    break
                action -= self.actions_per_intersection[ts]
        return actions

    def _compute_info(self):
        info = {}
        for ts in self.intersection_ids:
            info[f'{ts}_queue'] = self.data_collectors[ts].get_total_queued()
            info[f'{ts}_speed'] = self.data_collectors[ts].get_average_speed()
        return info

    def close(self):
        # Close any open connections or resources
        for ts in self.intersection_ids:
            self.signal_controllers[ts].close()  # Assuming SignalController has a close method

    def render(self, mode='human'):
        # This method could be implemented to visualize the current state of the intersections
        print("Current simulation time:", time.time() - self.start_time)
        for ts in self.intersection_ids:
            print(f"Intersection {ts}:")
            print(f"  Current phase: {self.signal_controllers[ts].get_current_phase()}")
            print(f"  Total queued: {self.data_collectors[ts].get_total_queued()}")
            print(f"  Average speed: {self.data_collectors[ts].get_average_speed()}")
        print("\n")
