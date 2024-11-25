from dataclasses import dataclass

from atscui.config.base_config import TrainingConfig


@dataclass
class DQNConfig(TrainingConfig):
    learning_rate: float = 1e-3
    learning_starts: int = 0
    train_freq: int = 1
    target_update_interval: int = 1000
    exploration_initial_eps: float = 0.05
    exploration_final_eps: float = 0.01

@dataclass
class PPOConfig(TrainingConfig):
    learning_rate: float = 1e-3
    n_steps: int = 1024
