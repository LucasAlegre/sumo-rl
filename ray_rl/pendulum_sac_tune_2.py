import argparse
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.sac import SACConfig
import os
from typing import Optional, Tuple, Dict, Any
import torch

# 获取当前工作目录的绝对路径
CURRENT_DIR = os.path.abspath(os.getcwd())


class TrainingConfig:
    def __init__(
            self,
            iterations: int = 20,  # tune 训练轮次
            num_episodes: int = 200,  # 每轮次的回合数
            eval_interval: int = 5,  # 评估间隔轮次
            checkpoint_path: Optional[str] = None,
            model_path: Optional[str] = None,
            num_workers: int = 2,  # worker数量
            buffer_size: int = 50000,  # 经验回放缓冲区大小
            train_batch_size: int = 256,  # 训练批次大小
            learning_rate: float = 3e-4,  # 学习率
            gamma: float = 0.99,  # 折扣因子
            tau: float = 0.005,  # 目标网络软更新系数
            alpha: float = 0.2,  # 熵正则化系数
            target_entropy: Optional[float] = None,  # 目标熵
            n_step: int = 1,  # n步回报
            exp_name: Optional[str] = None,  # 实验名称
            try_render: bool = False
    ):
        self.iterations = iterations
        self.num_episodes = num_episodes
        self.eval_interval = eval_interval
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_entropy = target_entropy
        self.n_step = n_step
        self.exp_name = exp_name
        self.try_render = try_render


def get_sac_config(config: TrainingConfig) -> SACConfig:
    """
    获取SAC配置
    """
    num_gpus = 1 if torch.cuda.is_available() else 0

    # 使用Pendulum-v1作为连续动作空间的示例环境
    base_config = (
        SACConfig()
        .environment("Pendulum-v1")
        .framework("torch")
        .env_runners(
            num_env_runners=config.num_workers,
        )
        .training(
            train_batch_size=config.train_batch_size,
            actor_lr=config.learning_rate,
            critic_lr=config.learning_rate,
            alpha_lr=config.learning_rate,
            gamma=config.gamma,
            tau=config.tau,
            initial_alpha=config.alpha,
            n_step=config.n_step,
            replay_buffer_config={
                "replay_batch_size": config.buffer_size,
            }
        )
        .resources(num_gpus=num_gpus)
    )

    # Tune搜索空间
    base_config.training(
        optimization_config={
            "actor_learning_rate": tune.loguniform(1e-4, 1e-3),
            "critic_learning_rate": tune.loguniform(1e-4, 1e-3),
            "entropy_learning_rate": tune.loguniform(1e-4, 1e-3)
        },
        tau=tune.uniform(0.001, 0.01),
        initial_alpha=tune.uniform(0.1, 0.5),
    )

    return base_config


def train_sac_with_tune(config: TrainingConfig) -> Tuple[Optional[str], Optional[Optional[str]]]:
    """
    使用Tune训练SAC，支持从checkpoint继续训练
    """
    local_dir = os.path.join(CURRENT_DIR, "ray_results")
    os.makedirs(local_dir, exist_ok=True)

    tune_config = get_sac_config(config)

    print("==========config.checkpoint_tune:", config.checkpoint_path)
    # 如果提供了checkpoint，设置restore参数
    restore_path = config.checkpoint_path if config.checkpoint_path and os.path.exists(config.checkpoint_path) else None

    analysis = tune.run(
        "SAC",
        config=tune_config.to_dict(),
        stop={"training_iteration": config.iterations},
        num_samples=4,
        metric="env_runners/episode_reward_mean",
        mode="max",
        storage_path=local_dir,
        checkpoint_at_end=True,
        name=config.exp_name,
        restore=restore_path,
        verbose=1,
    )

    best_trial = analysis.best_trial

    if best_trial:
        best_checkpoint = analysis.best_checkpoint
        if best_checkpoint:
            # 1. 保存完整checkpoint (包含所有训练状态)
            best_checkpoint_path = best_checkpoint.path
            best_config = best_trial.config
            print("Best config:", best_config)

            # 2. 创建新的SAC实例并加载最佳checkpoint
            save_dir = os.path.join(local_dir, "best_pendulum_sac_tune")
            os.makedirs(save_dir, exist_ok=True)
            best_algo = SAC(config=best_config)
            best_algo.restore(best_checkpoint_path)

            # 3. 分别保存deployment模型和完整checkpoint
            # 3.1 保存完整checkpoint（用于继续训练）
            checkpoint_path = os.path.join(save_dir, "full_checkpoint")
            os.makedirs(checkpoint_path, exist_ok=True)
            best_algo.save_checkpoint(checkpoint_path)
            print(f"Full checkpoint saved to: {checkpoint_path}")

            # 3.2 保存deployment模型（用于部署）
            model_path = os.path.join(save_dir, "deployment_model.pkl")
            best_algo.save(model_path)
            print(f"Deployment model saved to: {model_path}")

            return checkpoint_path, model_path

    return None, None


def load_model(model_path: str, config: Dict[str, Any]) -> SAC:
    algo = SAC(config=config)

    # 加载deployment模型（仅包含模型权重）
    algo.restore(model_path)

    return algo


def load_checkpoint(checkpoint_path: str, config: Dict[str, Any]) -> SAC:
    algo = SAC(config=config)

    # 加载完整checkpoint（包含训练状态）
    algo.load_checkpoint(checkpoint_path)

    return algo


def evaluate_sac(algo: SAC, num_episodes: int = 10) -> float:
    """
    评估SAC智能体的性能
    """
    env = gym.make("Pendulum-v1")
    total_rewards = []

    for _ in range(num_episodes):
        episode_reward = 0
        obs, _ = env.reset()
        done = False
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    print(f"Evaluation over {num_episodes} episodes: {mean_reward:.2f}")
    return mean_reward


def inference_sac(algo: SAC, num_episodes: int = 5, try_render: bool = True) -> list:
    """
    运行SAC智能体进行推理
    """
    env = gym.make("Pendulum-v1", render_mode="human" if try_render else None)
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Steps = {step_count}, Reward = {total_reward}")

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nInference Statistics:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Max reward: {max(episode_rewards)}")
    print(f"Min reward: {min(episode_rewards)}")

    return episode_rewards


def save_metrics(metrics: list, filename: str = "performance_metrics.txt"):
    """
    保存性能指标到文件
    """
    filepath = os.path.join(CURRENT_DIR, filename)
    with open(filepath, "w") as f:
        f.write("Performance Metrics:\n")
        f.write(f"Number of episodes: {len(metrics)}\n")
        f.write(f"Mean reward: {np.mean(metrics):.2f}\n")
        f.write(f"Standard deviation: {np.std(metrics):.2f}\n")
        f.write(f"Max reward: {max(metrics)}\n")
        f.write(f"Min reward: {min(metrics)}\n")
        f.write("\nDetailed records:\n")
        for i, reward in enumerate(metrics, 1):
            f.write(f"Episode {i}: {reward}\n")
    print(f"Metrics saved to: {filepath}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用Ray Tune进行SAC算法实验')
    parser.add_argument('--operation', type=str, choices=['TRAIN', 'INFERENCE'],
                        required=True, help='操作类型: TRAIN-训练, INFERENCE-推理')
    parser.add_argument('--iterations', type=int, default=20, help='tune训练轮次')
    parser.add_argument('--checkpoint-path', type=str, help='tune测试时的检查点路径')
    parser.add_argument('--model-path', type=str, help='tune推理时的模型路径')
    parser.add_argument('--exp-name', type=str, default='pendulum_sac_tune', help='实验名称')
    parser.add_argument('--eval-interval', type=int, default=5, help='评估间隔')
    parser.add_argument('--try-render', action='store_true', help='测试时是否渲染环境')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    ray.init()

    # 创建训练配置
    config = TrainingConfig(
        iterations=args.iterations,
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        eval_interval=args.eval_interval,
        exp_name=args.exp_name,
        try_render=args.try_render,
    )

    if args.operation == 'TRAIN':
        # 使用Tune的训练
        print("\n\n==============================Training with Tune:")
        best_checkpoint_path, best_model_path = train_sac_with_tune(config)
        print("==============================best_checkpoint_path:", best_checkpoint_path)
        print("==============================best_model_path:", best_model_path)

    elif args.operation == 'INFERENCE':
        # 如果提供了tune检查点，加载并评估
        if args.model_path:
            print("\n==============================Loading tune model:")
            tune_config = get_sac_config(config)
            # algo = tune_config.build()
            # algo.restore(args.model_path)
            algo = load_model(args.model_path, tune_config.to_dict())
            # algo = load_checkpoint(args.model_path, tune_config.to_dict())
            print("===========================================================")
            performance_metrics = inference_sac(algo, num_episodes=10, try_render=config.try_render)
            save_metrics(performance_metrics, filename="performance_metrics_pendulum_sac_tune.txt")

    ray.shutdown()

"""
(1)Ubuntu - train:
 python ray_rl/pendulum_sac_tune_2.py --operation=TRAIN --iterations=500 \
 --checkpoint-path="/home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_tune/SAC_Pendulum-v1_f9bd5_00003_3_initial_alpha=0.3057,actor_learning_rate=0.0001,critic_learning_rate=0.0001,entropy_learning_rate=0._2024-11-04_13-52-28/checkpoint_000000" \

(2) Ubuntu - inference:


(3) Mac OS - train:
python ray_rl/pendulum_sac_tune_2.py --operation=TRAIN --iterations=40 \
--checkpoint-path="/Users/xnpeng/sumoptis/sumo-rl/ray_results/best_pendulum_sac_tune/full_checkpoint"

(4) Mac OS - inference:

python ray_rl/pendulum_sac_tune_2.py --operation=INFERENCE \
--model-path="/Users/xnpeng/sumoptis/sumo-rl/ray_results/best_pendulum_sac_tune/deployment_model.pkl"


"""
