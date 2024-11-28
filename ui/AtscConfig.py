from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    net_file: str
    rou_file: str
    num_seconds: int = 20000
    single_agent: bool = True
    gui: bool = False
    render_mode: Optional[str] = None


@dataclass
class TrainingConfig(BaseConfig):
    total_timesteps: int = 100_000
    tensorboard_logs: str = "logs"
    n_eval_episodes: int = 10


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


@dataclass
class A2CConfig(TrainingConfig):
    learning_rate: float = 1e-3
    n_steps: int = 1024


@dataclass
class SACConfig(TrainingConfig):
    learning_rate: float = 1e-3
    n_steps: int = 1024

