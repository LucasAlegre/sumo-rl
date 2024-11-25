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

