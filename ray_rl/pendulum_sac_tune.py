import argparse
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.sac import SACConfig
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
            iterations_no_tune: int = 20,  # 训练轮次
            iterations_tune: int = 20,  # tune 训练轮次
            num_episodes: int = 200,  # 每轮次的回合数
            eval_interval: int = 5,  # 评估间隔轮次
            checkpoint_tune: Optional[str] = None,  # 加载的checkpoint with tune路径
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
        self.iterations_tune = iterations_tune
        self.num_episodes = num_episodes
        self.eval_interval = eval_interval
        self.checkpoint_tune = checkpoint_tune
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


def get_sac_config(config: TrainingConfig, use_tune: bool = True) -> SACConfig:
    """
    获取SAC配置
    """
    num_gpus = 1 if torch.cuda.is_available() else 0

    # 使用Pendulum-v1作为连续动作空间的示例环境
    base_config = (
        SACConfig()
        .environment("Pendulum-v1")
        .framework("torch")
        .rollouts(
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
                "replay_batch_size":config.buffer_size,
            }
        )
        .resources(num_gpus=num_gpus)
    )

    if use_tune:
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


def train_sac(config: TrainingConfig) -> Tuple[SAC, Optional[str]]:
    """
    训练SAC智能体，支持从checkpoint继续训练
    """
    local_dir = os.path.join(CURRENT_DIR, "ray_results")
    checkpoint_dir = os.path.join(local_dir, "pendulum_sac_no_tune")
    os.makedirs(checkpoint_dir, exist_ok=True)

    sac_config = get_sac_config(config, use_tune=False)

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


def train_sac_with_tune(config: TrainingConfig) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    使用Tune训练SAC，支持从checkpoint继续训练
    """
    local_dir = os.path.join(CURRENT_DIR, "ray_results")
    os.makedirs(local_dir, exist_ok=True)

    tune_config = get_sac_config(config, use_tune=True)

    print("==========config.checkpoint_tune:", config.checkpoint_tune)
    # 如果提供了checkpoint，设置restore参数
    restore_path = config.checkpoint_tune if config.checkpoint_tune and os.path.exists(config.checkpoint_tune) else None

    analysis = tune.run(
        "SAC",
        config=tune_config.to_dict(),
        stop={"training_iteration": config.iterations_tune},
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
            print("tune::run:Best hyperparameters:", best_config)
            return best_checkpoint_path, best_config

    return None, None


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
    parser.add_argument('--iterations-tune', type=int, default=20, help='tune训练轮次')
    parser.add_argument('--iterations-no-tune', type=int, default=10, help='no tune训练轮次')
    parser.add_argument('--checkpoint-tune', type=str, help='tune测试时的检查点路径')
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
        iterations_tune=args.iterations_tune,
        iterations_no_tune=args.iterations_no_tune,
        checkpoint_tune=args.checkpoint_tune,
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
        if algo is not None:
            performance_metrics = inference_sac(algo, num_episodes=10, try_render=config.try_render)
            save_metrics(performance_metrics, filename="performance_metrics_pendulum_sac_no_tune.txt")

        # 使用Tune的训练
        print("\n\n==============================Training with Tune:")
        best_checkpoint_tune, best_config = train_sac_with_tune(config)
        print("==============================best_checkpoint_tune:", best_checkpoint_tune)

        # 加载和评估最佳模型
        if best_checkpoint_tune and best_config:
            best_algo = SAC(config=best_config)
            best_algo.restore(best_checkpoint_tune)

            performance_metrics = inference_sac(best_algo, num_episodes=10, try_render=config.try_render)
            save_metrics(performance_metrics, filename="performance_metrics_pendulum_sac_tune.txt")
        else:
            print("No best checkpoint found from Tune training")
    
    elif args.operation == 'INFERENCE':
        # 检查是否提供了检查点路径
        if not (args.checkpoint_tune or args.checkpoint_no_tune):
            raise ValueError("推理模式需要提供至少一个检查点路径 (--checkpoint-tune 或 --checkpoint-no-tune)")

        # 如果提供了no-tune检查点，加载并评估
        if args.checkpoint_no_tune:
            print("\n==============================Loading no-tune checkpoint:")
            sac_config = get_sac_config(config, use_tune=False)
            algo = sac_config.build()
            algo.restore(args.checkpoint_no_tune)
            performance_metrics = inference_sac(algo, num_episodes=10, try_render=config.try_render)
            save_metrics(performance_metrics, filename="performance_metrics_pendulum_sac_no_tune.txt")

        # 如果提供了tune检查点，加载并评估
        if args.checkpoint_tune:
            print("\n==============================Loading tune checkpoint:")
            tune_config = get_sac_config(config, use_tune=True).to_dict()
            algo = SAC(config=tune_config)
            algo.restore(args.checkpoint_tune)
            performance_metrics = inference_sac(algo, num_episodes=10, try_render=config.try_render)
            save_metrics(performance_metrics, filename="performance_metrics_pendulum_sac_tune.txt")

    ray.shutdown()

"""
(1)Ubuntu:
 python ray_rl/pendulum_sac_tune.py --iterations-tune=400 --iterations-no-tune=50 \
 --checkpoint-tune="/home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_tune/SAC_Pendulum-v1_b41ec_00002_2_initial_alpha=0.2049,actor_learning_rate=0.0003,critic_learning_rate=0.0005,entropy_learning_rate=0._2024-11-04_09-11-21/checkpoint_000000" \
 --checkpoint-no-tune="/home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_no_tune/checkpoint_80"

--no-tune:
checkpoint_no_tune: /home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_no_tune/checkpoint_5
Episode 1: Steps = 200, Reward = -1.479035394261019
Episode 2: Steps = 200, Reward = -884.9222940178998
Episode 3: Steps = 200, Reward = -888.6182699151921
Episode 4: Steps = 200, Reward = -769.3482924883454
Episode 5: Steps = 200, Reward = -0.9325495365174847
Episode 6: Steps = 200, Reward = -897.7704440652453
Episode 7: Steps = 200, Reward = -770.8271331440368
Episode 8: Steps = 200, Reward = -738.1726931594903
Episode 9: Steps = 200, Reward = -916.6450709755005
Episode 10: Steps = 200, Reward = -871.9310219517179

Inference Statistics:
Mean reward: -674.06 ± 341.55
Max reward: -0.9325495365174847
Min reward: -916.6450709755005
Metrics saved to: /home/kemove/Projects/sumo-rl/performance_metrics_pendulum_sac_no_tune.txt

--tune
checkpoint-tune="/home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_tune/SAC_Pendulum-v1_b41ec_00002_2_initial_alpha=0.2049,actor_learning_rate=0.0003,critic_learning_rate=0.0005,entropy_learning_rate=0._2024-11-04_09-11-21/checkpoint_000000"
Episode 1: Steps = 200, Reward = -129.8839824417047
Episode 2: Steps = 200, Reward = -123.1287426688013
Episode 3: Steps = 200, Reward = -130.76487668182077
Episode 4: Steps = 200, Reward = -291.7253349382669
Episode 5: Steps = 200, Reward = -369.172969862427
Episode 6: Steps = 200, Reward = -2.5119956045702407
Episode 7: Steps = 200, Reward = -122.13189700443708
Episode 8: Steps = 200, Reward = -247.90041723507747
Episode 9: Steps = 200, Reward = -127.7395137407737
Episode 10: Steps = 200, Reward = -126.9328233794857

Inference Statistics:
Mean reward: -167.19 ± 99.90
Max reward: -2.5119956045702407
Min reward: -369.172969862427
Metrics saved to: /home/kemove/Projects/sumo-rl/performance_metrics_pendulum_sac_tune.txt

checkpoint-tune="/home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_tune/SAC_Pendulum-v1_a716f_00002_2_initial_alpha=0.3671,actor_learning_rate=0.0002,critic_learning_rate=0.0008,entropy_learning_rate=0._2024-11-04_10-01-06/checkpoint_000000"
Episode 1: Steps = 200, Reward = -367.60862384928106
Episode 2: Steps = 200, Reward = -871.7382073043447
Episode 3: Steps = 200, Reward = -780.6702178597513
Episode 4: Steps = 200, Reward = -706.182844414461
Episode 5: Steps = 200, Reward = -524.3347096208906
Episode 6: Steps = 200, Reward = -901.2835488892741
Episode 7: Steps = 200, Reward = -662.6300159435667
Episode 8: Steps = 200, Reward = -775.1062328180774
Episode 9: Steps = 200, Reward = -774.3495687709152
Episode 10: Steps = 200, Reward = -916.9484087079585

Inference Statistics:
Mean reward: -728.09 ± 164.21
Max reward: -367.60862384928106
Min reward: -916.9484087079585
Metrics saved to: /home/kemove/Projects/sumo-rl/performance_metrics_pendulum_sac_tune.txt


(2) Mac OS inference:

python ray_rl/pendulum_sac_tune.py --operation=INFERENCE \
 --checkpoint-tune="/Users/xnpeng/sumoptis/sumo-rl/ray_results/checkpoint_000000" \
 --checkpoint-no-tune="/Users/xnpeng/sumoptis/sumo-rl/ray_results/checkpoint_5"


(3) Ubuntu inference:

python ray_rl/pendulum_sac_tune.py --operation=INFERENCE \
 --checkpoint-tune="/home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_tune/SAC_Pendulum-v1_b41ec_00002_2_initial_alpha=0.2049,actor_learning_rate=0.0003,critic_learning_rate=0.0005,entropy_learning_rate=0._2024-11-04_09-11-21/checkpoint_000000" \
 --checkpoint-no-tune="/home/kemove/Projects/sumo-rl/ray_results/pendulum_sac_no_tune"


"""