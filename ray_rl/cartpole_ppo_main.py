from datetime import datetime

import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from typing import Dict, Any
import os
import argparse
import torch


def get_ppo_config(use_gpu: bool = False) -> PPOConfig:
    """
    创建PPO算法的配置

    Args:
        use_gpu: 是否使用GPU
    """
    num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0
    return (
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .rollouts(num_rollout_workers=2)
        .resources(num_gpus=num_gpus)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,
        )
    )


def train(num_iterations: int = 20, use_gpu: bool = False, save_path: str = None) -> str:
    """
    训练PPO模型

    Args:
        num_iterations: 训练迭代次数
        use_gpu: 是否使用GPU
        save_path: 模型保存路径，如果为None则使用默认路径

    Returns:
        str: 模型检查点路径
    """
    # 创建配置和算法实例
    config = get_ppo_config(use_gpu)
    algo = config.build()

    # 训练循环
    for i in range(num_iterations):
        result = algo.train()

        # 打印训练信息
        print(f"第 {i} 轮训练")
        metrics = result["env_runners"]
        print(f"episode_reward_mean: {metrics['episode_reward_mean']}")
        print(f"episode_len_mean: {metrics['episode_len_mean']}")

        # 每5轮评估一次
        if i % 5 == 0:
            eval_results = algo.evaluate()
            print(f"评估结果: {eval_results['env_runners']['episode_reward_mean']}")

    storage_path = None
    if not save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"model_{timestamp}"
        storage_path = os.path.join(os.path.abspath("./ray_results/"), model_path)
    else:
        storage_path = os.path.join(os.path.abspath("./ray_results/"), save_path)

    # 保存模型
    checkpoint_dir = algo.save(storage_path)
    print(f"模型保存在: {checkpoint_dir.checkpoint.path}")

    # 清理算法实例
    algo.stop()

    return checkpoint_dir.checkpoint.path


def test(checkpoint_path: str, num_episodes: int = 5, render: bool = True, use_gpu: bool = False) -> Dict[int, float]:
    """
    测试已训练的PPO模型

    Args:
        checkpoint_path: 模型检查点路径
        num_episodes: 测试回合数
        render: 是否渲染环境
        use_gpu: 是否使用GPU

    Returns:
        Dict[int, float]: 每个回合的奖励
    """
    # 创建配置和算法实例
    config = get_ppo_config(use_gpu)
    algo = config.build()

    storage_path = None
    if checkpoint_path:
        storage_path = os.path.join(os.path.abspath("./ray_results/"), checkpoint_path)

    # 检查检查点路径是否存在
    if not os.path.exists(storage_path):
        raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")

    # 加载模型
    algo.restore(storage_path)

    # 创建环境
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # 运行测试回合
    episode_rewards = {}
    for episode in range(num_episodes):
        episode_reward = 0
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        episode_rewards[episode] = episode_reward
        print(f"回合 {episode + 1} 奖励: {episode_reward}")

    # 清理
    env.close()
    algo.stop()

    return episode_rewards


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='PPO算法训练和测试CartPole环境')
    parser.add_argument('mode', choices=['train', 'test'], help='运行模式：train或test')
    parser.add_argument('--gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--iterations', type=int, default=20, help='训练迭代次数')
    parser.add_argument('--episodes', type=int, default=5, help='测试回合数')
    parser.add_argument('--model-path', type=str, help='模型路径：训练模式下为保存路径，测试模式下为加载路径')
    parser.add_argument('--no-render', action='store_true', help='测试时不渲染环境')
    return parser.parse_args()


def main():
    """
    主函数，根据命令行参数执行训练或测试
    """
    args = parse_arguments()

    # 检查GPU可用性
    if args.gpu and not torch.cuda.is_available():
        print("警告：GPU不可用，将使用CPU")
        args.gpu = False

    # 初始化Ray
    ray.init()

    try:
        if args.mode == 'train':
            print(f"开始训练... {'使用GPU' if args.gpu else '使用CPU'}")
            checkpoint_path = train(
                num_iterations=args.iterations,
                use_gpu=args.gpu,
                save_path=args.model_path
            )
            print(f"训练完成，模型保存在: {checkpoint_path}")

        else:  # test mode
            if not args.model_path:
                raise ValueError("测试模式需要指定模型路径 (--model-path)")

            print(f"开始测试... {'使用GPU' if args.gpu else '使用CPU'}")
            episode_rewards = test(
                checkpoint_path=args.model_path,
                num_episodes=args.episodes,
                render=not args.no_render,
                use_gpu=args.gpu
            )

            # 输出测试统计
            avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
            print(f"\n测试结果 - 平均奖励: {avg_reward:.2f}")

    finally:
        # 清理Ray
        ray.shutdown()


if __name__ == "__main__":
    main()

"""
(1)train
python ray_rl/cartpole_ppo_main.py train --model-path=cartpole_ppo_mac_cpu --gpu


(2)test:
python ray_rl/cartpole_ppo_main.py test --model-path=cartpole_ppo_mac_cpu --gpu --no-render

"""