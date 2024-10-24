import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.train import Checkpoint
from ray.tune.registry import register_env
from ray import tune
import ray
from typing import Dict, Any
import os
import argparse
import torch
from ray.air.config import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import json


def env_creator(env_config):
    """创建环境的工厂函数"""
    return gym.make("CartPole-v1")


def get_ppo_config(use_gpu: bool = False) -> Dict[str, Any]:
    """
    创建PPO算法的配置字典

    Args:
        use_gpu: 是否使用GPU
    """
    num_gpus = 1 if use_gpu and torch.cuda.is_available() else 0

    return {
        "env": "CartPole-v1",
        "framework": "torch",
        "num_gpus": num_gpus,
        "num_workers": 2,
        "train_batch_size": tune.choice([2000, 4000, 8000]),
        "sgd_minibatch_size": tune.choice([128, 256, 512]),
        "num_sgd_iter": tune.choice([10, 20, 30]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.uniform(0.9, 0.999),
        "lambda": tune.uniform(0.9, 1.0),
        "clip_param": tune.uniform(0.1, 0.3),
        "vf_clip_param": tune.uniform(5, 15),
        "model": {
            "fcnet_hiddens": tune.choice([
                [64, 64],
                [128, 128],
                [256, 256],
            ]),
        },
    }


def train_tune(num_samples: int = 10,
               max_training_iterations: int = 20,
               use_gpu: bool = False,
               exp_name: str = "ppo_cartpole",
               local_dir: str = "./ray_results") -> str:
    """
    使用Ray Tune训练PPO模型

    Args:
        num_samples: 采样的超参数组合数量
        max_training_iterations: 最大训练迭代次数
        use_gpu: 是否使用GPU
        exp_name: 实验名称
        local_dir: 结果保存目录

    Returns:
        str: 最佳检查点路径
    """
    # 注册环境
    register_env("CartPole-v1", env_creator)

    # 创建调度器
    scheduler = ASHAScheduler(
        max_t=max_training_iterations,
        grace_period=5,
        reduction_factor=2
    )

    # 创建搜索算法
    search_alg = HyperOptSearch()

    # 获取配置
    config = get_ppo_config(use_gpu)

    # 运行实验
    tuner = tune.Tuner(
        "PPO",
        run_config=RunConfig(
            name=exp_name,
            local_dir=local_dir,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True
            ),
        ),
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
    )

    results = tuner.fit()

    # 获取最佳试验结果
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # 保存最佳超参数
    best_config = best_result.config
    best_config_path = os.path.join(local_dir, exp_name, "best_config.json")
    os.makedirs(os.path.dirname(best_config_path), exist_ok=True)
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=4)

    print(f"最佳训练结果:")
    print(f"平均奖励: {best_result.metrics.get('episode_reward_mean')}")
    print(f"最佳配置保存在: {best_config_path}")

    return best_result.checkpoint


def test_from_checkpoint(checkpoint: Checkpoint,
                         num_episodes: int = 5,
                         render: bool = True,
                         use_gpu: bool = False) -> Dict[int, float]:
    """
    从检查点加载并测试模型

    Args:
        checkpoint: Ray Tune检查点
        num_episodes: 测试回合数
        render: 是否渲染环境
        use_gpu: 是否使用GPU

    Returns:
        Dict[int, float]: 每个回合的奖励
    """
    # 加载配置
    with open(os.path.join(os.path.dirname(checkpoint.path), "../best_config.json"), 'r') as f:
        config = json.load(f)

    # 更新GPU设置
    config["num_gpus"] = 1 if use_gpu and torch.cuda.is_available() else 0

    # 创建算法实例
    algo = PPOConfig().from_dict(config).build()

    # 加载检查点
    algo.restore(checkpoint.path)

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
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用Ray Tune进行PPO算法实验')
    parser.add_argument('mode', choices=['train', 'test'], help='运行模式：train或test')
    parser.add_argument('--gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--num-samples', type=int, default=10, help='超参数采样数量')
    parser.add_argument('--max-iterations', type=int, default=20, help='最大训练迭代次数')
    parser.add_argument('--episodes', type=int, default=5, help='测试回合数')
    parser.add_argument('--exp-name', type=str, default='ppo_cartpole', help='实验名称')
    parser.add_argument('--checkpoint-path', type=str, help='测试时的检查点路径')
    parser.add_argument('--no-render', action='store_true', help='测试时不渲染环境')
    return parser.parse_args()


def main():
    """主函数，根据命令行参数执行训练或测试"""
    args = parse_arguments()

    # 检查GPU可用性
    if args.gpu and not torch.cuda.is_available():
        print("警告：GPU不可用，将使用CPU")
        args.gpu = False

    # 初始化Ray
    ray.init()

    try:
        if args.mode == 'train':
            print(f"开始Ray Tune实验... {'使用GPU' if args.gpu else '使用CPU'}")
            checkpoint = train_tune(
                num_samples=args.num_samples,
                max_training_iterations=args.max_iterations,
                use_gpu=args.gpu,
                exp_name=args.exp_name
            )
            print(f"实验完成，最佳检查点保存在: {checkpoint.path}")

        else:  # test mode
            if not args.checkpoint_path:
                raise ValueError("测试模式需要指定检查点路径 (--checkpoint-path)")

            checkpoint = Checkpoint(args.checkpoint_path)
            print(f"开始测试... {'使用GPU' if args.gpu else '使用CPU'}")
            episode_rewards = test_from_checkpoint(
                checkpoint=checkpoint,
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

"""