from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import SAC


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


class A2CAgent(BaseAgent):
    def _create_model(self):
        return A2C(
            env=self.env,
            policy="MlpPolicy",
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
            tensorboard_log=self.config.tensorboard_logs,
            verbose=1
        )


class DQNAgent(BaseAgent):
    def _create_model(self):
        return DQN(
            env=self.env,
            policy="MlpPolicy",
            learning_rate=self.config.learning_rate,
            learning_starts=self.config.learning_starts,
            train_freq=self.config.train_freq,
            target_update_interval=self.config.target_update_interval,
            exploration_initial_eps=self.config.exploration_initial_eps,
            exploration_final_eps=self.config.exploration_final_eps,
            tensorboard_log=self.config.tensorboard_logs,
            verbose=1,
        )

    def train(self):
        return self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True
        )


class PPOAgent(BaseAgent):
    def _create_model(self):
        return PPO(
            env=self.env,
            policy="MlpPolicy",
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            tensorboard_log=self.config.tensorboard_logs,
            verbose=1
        )

    def train(self):
        return self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True
        )


class SACAgent(BaseAgent):
    def _create_model(self):
        return SAC(
            env=self.env,
            policy="MlpPolicy",
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            tensorboard_log=self.config.tensorboard_logs,
            verbose=1
        )
