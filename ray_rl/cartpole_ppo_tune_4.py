import argparse

import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import os
import warnings
from typing import Optional, Tuple, Dict, Any
import torch

# 获取当前工作目录的绝对路径
CURRENT_DIR = os.path.abspath(os.getcwd())


class TrainingConfig:
    def __init__(
            self,
            num_iterations: int = 20,  # 训练轮次
            tune_iterations: int = 20, # tune 训练轮次
            num_episodes: int = 200,  # 每轮训练的回合数
            eval_interval: int = 5,  # 评估间隔
            checkpoint_tune: Optional[str] = None,  # 加载的checkpoint with tune路径
            checkpoint_no_tune: Optional[str] = None,  # 加载的checkpoint no tune路径
            num_workers: int = 2,  # worker数量
            batch_size: int = 4000,  # 批次大小
            lr: float = 7e-5,  # 学习率
            gamma: float = 0.99,  # 折扣因子
            lambda_: float = 0.95,  # GAE参数
            entropy_coeff: float = 0.01,  # 熵系数
            clip_param: float = 0.1,  # PPO裁剪参数
            num_sgd_iter: int = 10,  # SGD迭代次数
            exp_name: Optional[str] = None,  # 实验名称
            try_render: bool = False
    ):
        self.num_iterations = num_iterations
        self.tune_iterations = tune_iterations
        self.num_episodes = num_episodes
        self.eval_interval = eval_interval
        self.checkpoint_tune = checkpoint_tune
        self.checkpoint_no_tune = checkpoint_no_tune
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coeff = entropy_coeff
        self.clip_param = clip_param
        self.num_sgd_iter = num_sgd_iter
        self.exp_name = exp_name
        self.try_render = try_render


def get_ppo_config(config: TrainingConfig, use_tune: bool = True) -> PPOConfig:
    """
    获取PPO配置
    """
    num_gpus = 1 if torch.cuda.is_available() else 0
    base_config = (
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .env_runners(
            num_env_runners=config.num_workers,
        )
        .training(
            train_batch_size=config.batch_size,
            lr=config.lr,
            gamma=config.gamma,
            lambda_=config.lambda_,
            entropy_coeff=config.entropy_coeff,
            clip_param=config.clip_param,
            num_sgd_iter=config.num_sgd_iter,
        )
        .resources(num_gpus=num_gpus)
    )

    if use_tune:
        # Tune搜索空间
        base_config.training(
            lr=tune.loguniform(1e-5, 1e-3),
            entropy_coeff=tune.uniform(0.0, 0.02),
            clip_param=tune.uniform(0.1, 0.3),
        )

    return base_config


def train_ppo(config: TrainingConfig) -> Tuple[PPO, Optional[str]]:
    """
    训练PPO智能体，支持从checkpoint继续训练
    """
    local_dir = os.path.join(CURRENT_DIR, "ray_results")
    checkpoint_dir = os.path.join(local_dir, "checkpoint_no_tune")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ppo_config = get_ppo_config(config, use_tune=False)

    print("==========config.checkpoint_no_tune:", config.checkpoint_no_tune)

    # 创建或加载算法
    if config.checkpoint_no_tune and os.path.exists(config.checkpoint_no_tune):
        print(f"Loading checkpoint from: {config.checkpoint_no_tune}")
        algo = ppo_config.build()
        algo.restore(config.checkpoint_no_tune)
        print("Successfully loaded checkpoint")
    else:
        print("Starting training from scratch")
        algo = ppo_config.build()

    best_reward = float('-inf')
    best_checkpoint_no_tune = None

    # 训练循环
    for i in range(config.num_iterations):
        result = algo.train()
        print(f"================algo.train::Iteration {i}: ", result['env_runners']['episode_reward_mean'])

        # 评估
        if i % config.eval_interval == 0:
            mean_reward = evaluate_ppo(algo)
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_checkpoint_no_tune = os.path.join(checkpoint_dir, f"checkpoint_{i}")
                algo.save(best_checkpoint_no_tune)
                print(f"New best model saved: {best_checkpoint_no_tune}")

    return algo, best_checkpoint_no_tune


def train_ppo_with_tune(config: TrainingConfig) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    使用Tune训练PPO，支持从checkpoint继续训练
    """
    local_dir = os.path.join(CURRENT_DIR, "ray_results")
    os.makedirs(local_dir, exist_ok=True)

    tune_config = get_ppo_config(config, use_tune=True)

    print("==========config.checkpoint_tune:",config.checkpoint_tune)
    # 如果提供了checkpoint，设置restore参数
    restore_path = config.checkpoint_tune if config.checkpoint_tune and os.path.exists(config.checkpoint_tune) else None

    analysis = tune.run(
        "PPO",
        config=tune_config.to_dict(),
        stop={"training_iteration": config.tune_iterations},
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
            best_checkpoint_path = best_checkpoint.path
            best_config = best_trial.config
            print("tune::run:Best hyperparameters:best_config[num_env_runners]", best_config["num_env_runners"])
            return best_checkpoint_path, best_config

    return None, None


def evaluate_ppo(algo: PPO, num_episodes: int = 10) -> float:
    """
    评估PPO智能体的性能
    """
    env = gym.make("CartPole-v1")
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


def inference_ppo(algo: PPO, num_episodes: int = 5, try_render: bool = True) -> list:
    """
    运行PPO智能体进行推理
    """
    env = gym.make("CartPole-v1", render_mode="human" if try_render else None)
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
    parser = argparse.ArgumentParser(description='使用Ray Tune进行PPO算法实验')
    parser.add_argument('--num-iterations', type=int, default=10, help='no tune训练轮次')
    parser.add_argument('--tune-iterations', type=int, default=20, help='tune训练轮次')
    parser.add_argument('--exp-name', type=str, default='checkpoint_tune', help='实验名称')
    parser.add_argument('--checkpoint-tune', type=str, help='tune测试时的检查点路径')
    parser.add_argument('--checkpoint-no-tune', type=str, help='no tune测试时的检查点路径')
    parser.add_argument('--eval-interval', type=int, default=5, help='评估间隔')
    parser.add_argument('--try-render', action='store_true', help='测试时不渲染环境')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    ray.init()

    # 创建训练配置
    config = TrainingConfig(
        num_iterations=args.num_iterations,  # 训练轮次
        tune_iterations=args.tune_iterations,
        eval_interval=args.eval_interval,  # 评估间隔
        exp_name=args.exp_name,
        try_render=args.try_render,
        checkpoint_tune=args.checkpoint_tune,  # 如果存在，从此checkpoint继续训练
        checkpoint_no_tune=args.checkpoint_no_tune  # 如果存在，从此checkpoint继续训练
    )

    # 不使用Tune的训练
    print("\n\n==============================Training without Tune:")
    algo, checkpoint_no_tune = train_ppo(config)
    print("==============================checkpoint_no_tune:", checkpoint_no_tune)

    # 加载和评估模型
    if algo is not None:
        performance_metrics = inference_ppo(algo,num_episodes=10, try_render=config.try_render)
        save_metrics(performance_metrics, filename="performance_metrics_no_tune.txt")

    # 使用Tune的训练
    print("\n\n==============================Training with Tune:")
    best_checkpoint_tune, best_config = train_ppo_with_tune(config)
    print("==============================best_checkpoint_tune:", best_checkpoint_tune)

    # 加载和评估最佳模型
    if best_checkpoint_tune and best_config:
        best_algo = PPO(config=best_config)
        best_algo.restore(best_checkpoint_tune)

        performance_metrics = inference_ppo(best_algo, num_episodes=10, try_render=config.try_render)
        save_metrics(performance_metrics, filename="performance_metrics_tune.txt")
    else:
        print("No best checkpoint found from Tune training")

    ray.shutdown()


"""
程序设计要求：
（1）使用两种框架训练,非tune方法 和 tune方法，保存训练结果：最佳模型 和 最侍检查点；并从检查点继续训练；
（2）使用GPU训练的智能体，部署在CPU机器上推理；
（3）在无图形显示的情况下，使用数据记录推理结果；

实验结果：
（1）no tune 训练结果 比 tune训练结果明显差；
（2）GPU训练结果 可以在 CPU上使用。

继续测试：
（1）并发实验parallel_env
（2）多智能体多环境交通环境RESCO的训练

"""
"""
训练
python ray_rl/cartpole_ppo_tune_4.py

ubuntu: python ray_rl/cartpole_ppo_tune_4.py --checkpoint-tune="/home/kemove/Projects/sumo-rl/ray_results/ppo_cartpole/PPO_CartPole-v1_b69e9_00001_1_clip_param=0.1510,entropy_coeff=0.0109,lr=0.0003_2024-10-29_10-58-45/checkpoint_000000" --checkpoint-tune="/home/kemove/Projects/sumo-rl/ray_results/checkpoint_no_tune/checkpoint_40" --num-iterations=50


(1)非tune训练
checkpoint_no_tune: /Users/xnpeng/sumoptis/sumo-rl/ray_results/checkpoint_no_tune/checkpoint_5

Ubuntu:/home/kemove/Projects/sumo-rl/ray_results/checkpoint_no_tune/checkpoint_40
Episode 1: Steps = 455, Reward = 455.0
Episode 2: Steps = 406, Reward = 406.0
Episode 3: Steps = 500, Reward = 500.0
Episode 4: Steps = 295, Reward = 295.0
Episode 5: Steps = 341, Reward = 341.0
Episode 6: Steps = 231, Reward = 231.0
Episode 7: Steps = 382, Reward = 382.0
Episode 8: Steps = 460, Reward = 460.0
Episode 9: Steps = 371, Reward = 371.0
Episode 10: Steps = 319, Reward = 319.0

Inference Statistics:
Mean reward: 376.00 ± 78.55
Max reward: 500.0
Min reward: 231.0

(2)tune训练
MacOS:/Users/xnpeng/sumoptis/sumo-rl/ray_results/PPO_2024-10-28_17-11-56/PPO_CartPole-v1_ae002_00000_0_clip_param=0.1402,entropy_coeff=0.0006,lr=0.0004_2024-10-28_17-11-56/checkpoint_000000

Ubuntu:
/home/kemove/Projects/sumo-rl/ray_results/ppo_cartpole/PPO_CartPole-v1_b69e9_00001_1_clip_param=0.1510,entropy_coeff=0.0109,lr=0.0003_2024-10-29_10-58-45/checkpoint_000000
/home/kemove/Projects/sumo-rl/ray_results/ppo_cartpole/PPO_CartPole-v1_d15d3_00001_1_clip_param=0.2020,entropy_coeff=0.0122,lr=0.0003_2024-10-29_12-03-56/checkpoint_000000
/home/kemove/Projects/sumo-rl/ray_results/checkpoint_tune/PPO_CartPole-v1_b64b7_00003_3_clip_param=0.2767,entropy_coeff=0.0190,lr=0.0005_2024-10-29_15-52-14/checkpoint_000000

Episode 1: Steps = 500, Reward = 500.0
Episode 2: Steps = 500, Reward = 500.0
Episode 3: Steps = 500, Reward = 500.0
Episode 4: Steps = 500, Reward = 500.0
Episode 5: Steps = 500, Reward = 500.0
Episode 6: Steps = 500, Reward = 500.0
Episode 7: Steps = 500, Reward = 500.0
Episode 8: Steps = 500, Reward = 500.0
Episode 9: Steps = 500, Reward = 500.0
Episode 10: Steps = 500, Reward = 500.0

Inference Statistics:
Mean reward: 500.00 ± 0.00
Max reward: 500.0
Min reward: 500.0

"""