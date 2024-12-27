from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    net_file: str
    rou_file: str
    csv_path: str
    model_path: str
    eval_path: str
    predict_path: str
    single_agent: bool = True
    gui: bool = False
    render_mode: Optional[str] = None
    operation: Optional[str] = "TRAIN"
    algo_name: Optional[str] = "DQN"


@dataclass
class TrainingConfig(BaseConfig):
    total_timesteps: int = 1_000_000  # 总训练时间步
    num_seconds: int = 20_000  # 每回合episode仿真步(时长)
    n_steps: int = 1024  # A2C价值网络更新间隔时间步
    n_eval_episodes: int = 10  # 评估回合数
    tensorboard_logs: str = "logs"
    learning_rate = 1e-3
    gamma = 0.9
    verbose: int = 0
    learning_starts: int = 0
    train_freq: int = 1
    target_update_interval: int = 1000
    exploration_initial_eps: float = 0.05
    exploration_final_eps: float = 0.01
    batch_size: int = 1024
    n_epochs: int = 100
    buffer_size: int = 10_000
    tau: float = 0.001

@dataclass
class RunningConfig(BaseConfig):
    operation = "PREDICT"
    gui: bool = False
    render_mode = None
    num_seconds = 100
