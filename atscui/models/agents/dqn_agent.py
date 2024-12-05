from typing import Tuple, Any

from stable_baselines3 import DQN

from atscui.models.base_agent import BaseAgent


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
            verbose=self.config.verbose,
        )

    def train(self):
        return self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True
        )
