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

        self.observation_spaces = {ts: self.traffic_signals[ts].observation_space for ts in self.intersection_ids}
        self.action_spaces = {ts: self.traffic_signals[ts].action_space for ts in self.intersection_ids}

        self.observation_space = gym.spaces.Dict({ts: self.observation_spaces[ts] for ts in self.intersection_ids})
        self.action_space = gym.spaces.Dict({ts: self.action_spaces[ts] for ts in self.intersection_ids})

    def reset(self):
        self.start_time = time.time()
        self.next_action_time = self.start_time

        # Reset all traffic signals and data collectors
        for ts in self.intersection_ids:
            self.traffic_signals[ts].signal_controller.set_phase(0)  # Set to initial phase
            self.data_collectors[ts].update()  # Ensure data is fresh

        observations = {ts: self.traffic_signals[ts].compute_observation() for ts in self.intersection_ids}
        return observations, {}  # Return initial observation and an empty info dict

    def step(self, actions):
        # Apply actions
        for ts, action in actions.items():
            if self.traffic_signals[ts].time_to_act:
                self.traffic_signals[ts].set_next_phase(action)

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

        # Check if simulation is done
        done = time.time() - self.start_time >= self.num_seconds
        terminated = done
        truncated = False  # We're not truncating episodes in this environment

        # Compute info
        info = self._compute_info()

        return observations, rewards, terminated, truncated, info

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