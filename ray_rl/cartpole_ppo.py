import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from typing import Dict, Any
import os


def get_ppo_config() -> PPOConfig:
    """
    创建PPO算法的配置
    """
    return (
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .rollouts(num_rollout_workers=2)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,
        )
    )


def train(num_iterations: int = 20) -> str:
    """
    训练PPO模型

    Args:
        num_iterations: 训练迭代次数

    Returns:
        str: 模型检查点路径
    """
    # 创建配置和算法实例
    config = get_ppo_config()
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

    # 保存模型
    storage_path = os.path.abspath("./ray_results/cartpole_ppo/")
    checkpoint_dir = algo.save(storage_path)
    print(f"模型保存在: {checkpoint_dir.checkpoint.path}")

    # 清理算法实例
    algo.stop()

    return checkpoint_dir.checkpoint.path


def test(checkpoint_path: str, num_episodes: int = 5, render: bool = True) -> Dict[int, float]:
    """
    测试已训练的PPO模型

    Args:
        checkpoint_path: 模型检查点路径
        num_episodes: 测试回合数
        render: 是否渲染环境

    Returns:
        Dict[int, float]: 每个回合的奖励
    """
    # 创建配置和算法实例
    config = get_ppo_config()
    algo = config.build()

    # 加载模型
    algo.restore(checkpoint_path)

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


def main():
    """
    主函数，用于训练和测试模型
    """
    # 初始化Ray
    ray.init()

    try:
        # 训练模型
        print("开始训练...")
        checkpoint_path = train(num_iterations=20)

        # 等待几秒，确保训练环境完全关闭
        import time
        time.sleep(2)

        # 测试模型
        print("\n开始测试...")
        episode_rewards = test(checkpoint_path, num_episodes=5, render=False)

        # 输出平均奖励
        avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
        print(f"\n测试结果 - 平均奖励: {avg_reward:.2f}")

    finally:
        # 清理Ray
        ray.shutdown()


if __name__ == "__main__":
    main()