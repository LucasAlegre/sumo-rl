from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseAgent(ABC):
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self):
        """Create and return the specific algorithm model"""
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Train the agent and return training metrics"""
        pass

    @abstractmethod
    def predict(self, observation) -> Tuple[Any, Any]:
        """Make a prediction given an observation"""
        pass

    def save(self, path: str):
        """Save the model to the specified path"""
        self.model.save(path)

    def load(self, path: str):
        """Load the model from the specified path"""
        self.model.load(path)