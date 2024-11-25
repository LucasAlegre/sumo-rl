import os

import gymnasium as gym
import numpy as np
import traci

from atscui.environment.base_env import BaseEnv


class SumoEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self._setup_sumo()
        self.action_space = gym.spaces.Discrete(4)  # 4 possible phases
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(16,), dtype=np.float32
        )

    def _setup_sumo(self):
        """Setup SUMO simulation"""
        if "SUMO_HOME" not in os.environ:
            raise RuntimeError("Please set SUMO_HOME environment variable")

        self.sumo_binary = "sumo" if not self.config.gui else "sumo-gui"
        self.sumo_cmd = [
            self.sumo_binary,
            "-n", self.config.net_file,
            "-r", self.config.rou_file,
            "--no-step-log",
            "--no-warnings",
        ]

    def step(self, action):
        # Implementation of one simulation step
        next_state = self._get_state()
        reward = self._calculate_reward()
        done = self._is_done()
        info = self._get_info()

        return next_state, reward, done, info

    def reset(self):
        traci.start(self.sumo_cmd)
        self.state = self._get_state()
        return self.state

    def close(self):
        if hasattr(self, 'traci') and self.traci.isLoaded():
            self.traci.close()
