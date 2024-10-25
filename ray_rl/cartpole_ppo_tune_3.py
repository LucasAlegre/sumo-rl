import warnings

import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import os
import torch

CURRENT_DIR = os.path.abspath(os.getcwd())


# 基础PPO配置
def get_ppo_config(use_tune=True):
    num_gpus = 1 if torch.cuda.is_available() else 0
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .rollouts(num_rollout_workers=2)
        .training(
            train_batch_size=4000,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.95,
            entropy_coeff=0.01,
            clip_param=0.2,
            num_sgd_iter=10,
        )
        .resources(num_gpus=num_gpus)
    )

    if use_tune:
        # Tune搜索空间
        config.training(
            lr=tune.loguniform(1e-5, 1e-3),
            entropy_coeff=tune.uniform(0.0, 0.02),
            clip_param=tune.uniform(0.1, 0.3),
        )

    return config


# 训练函数（不使用Tune）
def train_ppo():
    checkpoint_dir = os.path.join(CURRENT_DIR, "checkpoints")

    config = get_ppo_config(use_tune=False)
    algo = config.build()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_reward = float('-inf')
    best_checkpoint = None

    # 训练循环
    for i in range(10):
        result = algo.train()
        print(f"================algo.train::Iteration {i}: ", result['env_runners']['episode_reward_mean'])

        # 每5轮评估一次
        if i % 5 == 0:
            mean_reward = evaluate_ppo(algo)
            if mean_reward > best_reward:
                best_reward = mean_reward
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}")
                algo.save(checkpoint_path)
                best_checkpoint = checkpoint_path

    return algo, best_checkpoint


# 使用Tune进行训练
def train_ppo_with_tune():
    local_dir = os.path.join(CURRENT_DIR, "ray_results")
    os.makedirs(local_dir, exist_ok=True)

    config = get_ppo_config(use_tune=True)

    # 定义tune实验
    analysis = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 10},
        num_samples=4,  # 运行4次不同的超参数组合
        metric="env_runners/episode_reward_mean",
        mode="max",
        verbose=0,
        storage_path=local_dir,
        checkpoint_at_end=True  # 确保在训练结束时保存checkpoint
    )

    # 获取最佳试验结果
    best_trial = analysis.best_trial

    if best_trial:
        best_checkpoint = analysis.best_checkpoint
        if best_checkpoint:
            best_checkpoint_dir = best_checkpoint.path
            best_config = best_trial.config
            print("**************************tune::run:Best hyperparameters:num_env_runners", best_config["num_env_runners"])
            return best_checkpoint_dir, best_config

    return None, None


# 评估函数
def evaluate_ppo(algo, num_episodes=10):
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
    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@Evaluation over {num_episodes} episodes: {mean_reward:.2f}")
    return mean_reward


# 推理函数
def inference_ppo(algo, num_episodes=5, try_render=True):
    render_mode = None
    if try_render:
        try:
            # 尝试创建带渲染的环境
            test_env = gym.make("CartPole-v1", render_mode="human")
            test_env.close()
            render_mode = "human"
        except Exception as e:
            warnings.warn(f"无法创建渲染环境，将使用无渲染模式运行: {e}")
            render_mode = None

    env = gym.make("CartPole-v1", render_mode=render_mode)
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
        print(f"Episode {episode + 1}: 总步数 = {step_count}, 总奖励 = {total_reward}")

    env.close()

    # 打印统计信息
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n推理统计:")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"最高奖励: {max(episode_rewards)}")
    print(f"最低奖励: {min(episode_rewards)}")

    return episode_rewards


# 保存性能指标到文件
def save_metrics(metrics, filename="performance_metrics.txt"):
    filepath = os.path.join(CURRENT_DIR, filename)
    with open(filepath, "w") as f:
        f.write("性能指标统计:\n")
        f.write(f"样本数: {len(metrics)}\n")
        f.write(f"平均奖励: {np.mean(metrics):.2f}\n")
        f.write(f"标准差: {np.std(metrics):.2f}\n")
        f.write(f"最高奖励: {max(metrics)}\n")
        f.write(f"最低奖励: {min(metrics)}\n")
        f.write("\n详细记录:\n")
        for i, reward in enumerate(metrics, 1):
            f.write(f"Episode {i}: {reward}\n")
    print(f"指标已保存到: {filepath}")


if __name__ == "__main__":
    ray.init()

    # 不使用Tune的训练
    print("----------------------------Training without Tune:")
    algo, checkpoint_path = train_ppo()

    # 使用Tune的训练
    print("\n----------------------------Training with Tune:")
    best_checkpoint_dir, best_config = train_ppo_with_tune()

    # 加载不使用Tune训练的最佳检查点
    print("\n############################Evaluating model trained without Tune:")
    if checkpoint_path:
        algo.restore(checkpoint_path)
        evaluate_ppo(algo)
        # 进行推理演示
        print("\n========================Running inference with best model:")
        performance_metrics = inference_ppo(algo, num_episodes=10)
        # 保存性能指标
        save_metrics(performance_metrics, filename="performance_metrics_without_tune.txt")

    # 加载使用Tune训练的最佳检查点
    print("\n############################Evaluating model trained with Tune:")
    if best_checkpoint_dir and best_config:
        best_algo = PPO(config=best_config)
        best_algo.restore(best_checkpoint_dir)
        evaluate_ppo(best_algo)

        # 进行推理演示
        print("\n========================Running inference with best model:")
        performance_metrics = inference_ppo(best_algo, num_episodes=10)

        # 保存性能指标
        save_metrics(performance_metrics, filename="performance_metrics_tune.txt")
    else:
        print("No best checkpoint found from Tune training")

    ray.shutdown()
