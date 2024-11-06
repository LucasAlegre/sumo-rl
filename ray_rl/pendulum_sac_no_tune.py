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
            iterations_no_tune: int = 20,  # 训练轮次
            num_episodes: int = 200,  # 每轮次的回合数
            eval_interval: int = 5,  # 评估间隔轮次
            checkpoint_no_tune: Optional[str] = None,  # 加载的checkpoint no tune路径
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
        self.iterations_no_tune = iterations_no_tune
        self.num_episodes = num_episodes
        self.eval_interval = eval_interval
        self.checkpoint_no_tune = checkpoint_no_tune
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

    return base_config


def train_sac(config: TrainingConfig) -> Tuple[SAC, Optional[str]]:
    """
    训练SAC智能体，支持从checkpoint继续训练
    """
    local_dir = os.path.join(CURRENT_DIR, "ray_results")
    checkpoint_dir = os.path.join(local_dir, "pendulum_sac_no_tune")
    os.makedirs(checkpoint_dir, exist_ok=True)

    sac_config = get_sac_config(config)

    print("==========config.checkpoint_no_tune:", config.checkpoint_no_tune)

    # 创建或加载算法
    if config.checkpoint_no_tune and os.path.exists(config.checkpoint_no_tune):
        print(f"Loading checkpoint from: {config.checkpoint_no_tune}")
        algo = sac_config.build()
        algo.restore(config.checkpoint_no_tune)
        print("Successfully loaded checkpoint")
    else:
        print("Starting training from scratch")
        algo = sac_config.build()

    best_reward = float('-inf')
    best_checkpoint_no_tune = None

    # 训练循环
    for i in range(config.iterations_no_tune):
        result = algo.train()
        print(f"================algo.train::Iteration {i}: ", result['env_runners']['episode_reward_mean'])

        # 评估
        if i % config.eval_interval == 0:
            mean_reward = evaluate_sac(algo)
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_checkpoint_no_tune = os.path.join(checkpoint_dir, f"checkpoint_{i}")
                algo.save(best_checkpoint_no_tune)
                print(f"New best model saved: {best_checkpoint_no_tune}")

    return algo, best_checkpoint_no_tune


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
    parser.add_argument('--iterations-no-tune', type=int, default=10, help='no tune训练轮次')
    parser.add_argument('--checkpoint-no-tune', type=str, help='no tune测试时的检查点路径')
    parser.add_argument('--exp-name', type=str, default='pendulum_sac_tune', help='实验名称')
    parser.add_argument('--eval-interval', type=int, default=5, help='评估间隔')
    parser.add_argument('--try-render', action='store_true', help='测试时是否渲染环境')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    ray.init()

    # 创建训练配置
    config = TrainingConfig(
        iterations_no_tune=args.iterations_no_tune,
        checkpoint_no_tune=args.checkpoint_no_tune,
        eval_interval=args.eval_interval,
        exp_name=args.exp_name,
        try_render=args.try_render,
    )

    if args.operation == 'TRAIN':
        # 不使用Tune的训练
        print("\n\n==============================Training without Tune:")
        algo, checkpoint_no_tune = train_sac(config)
        print("==============================checkpoint_no_tune:", checkpoint_no_tune)

        # 加载和评估模型
        # if algo is not None:
        #     performance_metrics = inference_sac(algo, num_episodes=10, try_render=config.try_render)
        #     save_metrics(performance_metrics, filename="performance_metrics_pendulum_sac_no_tune.txt")

    elif args.operation == 'INFERENCE':
        # 检查是否提供了检查点路径
        if not args.checkpoint_no_tune:
            raise ValueError("推理模式需要提供一个检查点路径 (--checkpoint-no-tune)")

        # 如果提供了no-tune检查点，加载并评估
        if args.checkpoint_no_tune:
            print("\n==============================Loading no-tune checkpoint:")
            sac_config = get_sac_config(config)
            algo = sac_config.build()
            algo.restore(args.checkpoint_no_tune)
            performance_metrics = inference_sac(algo, num_episodes=10, try_render=config.try_render)
            save_metrics(performance_metrics, filename="performance_metrics_pendulum_sac_no_tune.txt")

    ray.shutdown()

"""
(1)Ubuntu - train:
 python ray_rl/pendulum_sac_no_tune.py --operation=TRAIN --iterations-no-tune=20 \
 --checkpoint-no-tune="/home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_no_tune/checkpoint_0"


(2) Ubuntu - inference:


(3) Mac OS - train:
python ray_rl/pendulum_sac_no_tune.py --operation=TRAIN --iterations-no-tune=20 \
--checkpoint-no-tune="/Users/xnpeng/sumoptis/sumo-rl/ray_results/pendulum_sac_no_tune/checkpoint_0" \


(4) Mac OS - inference:

python ray_rl/pendulum_sac_no_tune.py --operation=INFERENCE \
--checkpoint-no-tune="/Users/xnpeng/sumoptis/sumo-rl/ray_results/pendulum_sac_no_tune/checkpoint_0" \


"""
