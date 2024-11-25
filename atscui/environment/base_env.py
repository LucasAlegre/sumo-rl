from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import gym
import numpy as np


class BaseEnv(gym.Env, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state = None

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step within the environment"""
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment to an initial state"""
        pass

    @abstractmethod
    def close(self):
        """Clean up the environment"""
        pass