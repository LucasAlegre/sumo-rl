from stable_baselines3 import PPO
from atscui.models.base_agent import BaseAgent


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
            verbose=self.config.verbose,
        )

    def train(self):
        return self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True
        )
