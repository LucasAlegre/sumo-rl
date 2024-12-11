from stable_baselines3 import A2C

from atscui.models.base_agent import BaseAgent


class A2CAgent(BaseAgent):

    def _create_model(self):
        return A2C(
            env=self.env,
            policy="MlpPolicy",
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
            tensorboard_log=self.config.tensorboard_logs,
            verbose=self.config.verbose
        )

    def train(self):
        return self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True
        )