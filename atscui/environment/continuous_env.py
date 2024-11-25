import gymnasium as gym
import numpy as np

from atscui.environment.sumo_env import SumoEnv


class ContinuousSumoEnv(SumoEnv):
    def __init__(self, config):
        super().__init__(config)
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

    def step(self, action):
        # Convert continuous action to discrete traffic signal timing
        phase_duration = self._convert_action_to_duration(action)
        return super().step(phase_duration)

    def _convert_action_to_duration(self, action):
        # Convert continuous action to phase duration
        min_duration = 5
        max_duration = 60
        duration = min_duration + action[0] * (max_duration - min_duration)
        return int(duration)
