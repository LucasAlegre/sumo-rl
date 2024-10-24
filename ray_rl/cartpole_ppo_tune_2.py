import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


# 基础PPO配置
def get_ppo_config(use_tune=True):
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .rollouts(num_env_runners=2)
        .training(
            train_batch_size=4000,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.95,
            entropy_coeff=0.01,
            clip_param=0.2,
            num_sgd_iter=10,
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,
        )
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
    config = get_ppo_config(use_tune=False)
    algo = config.build()

    # 训练循环
    for i in range(20):
        result = algo.train()   # result: 一个描述训练过程的字典dict -> cartpole_ppo_tune_2_algo_train_result.txt
        print(f"================algo.train::Iteration {i}:", pretty_print(result))
        # 每5轮评估一次,两种评估方式有何区别？
        if i % 5 == 0:
            evaluate_ppo(algo)
            # eval_result = algo.evaluate()  # 并发方式评估
            # print(f"##########algo.evaluate(评估结果): {eval_result['env_runners']['episode_reward_mean']}")

    return algo


# 使用Tune进行训练
def train_ppo_with_tune():
    config = get_ppo_config(use_tune=True)

    # 定义tune实验
    analysis = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 20},
        num_samples=4,  # 运行4次不同的超参数组合
        metric="env_runners/episode_reward_mean",
        mode="max"
    )

    print("========== tune.run::analysis:", pretty_print(analysis.best_trial.last_result))

    # 获取最佳配置
    best_config = analysis.best_config
    best_checkpoint = analysis.best_checkpoint

    print("========== tune.run::Best hyperparameters:", pretty_print(best_config))
    return best_checkpoint


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
    print(f"##########evaluate_ppo(评估结果):{num_episodes} episodes: {mean_reward:.2f}")
    return mean_reward


# 推理函数
def inference_ppo(algo, render=True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if render:
            env.render()

    env.close()
    return total_reward


if __name__ == "__main__":
    ray.init()

    # 不使用Tune的训练
    print("Training without Tune:")
    # algo = train_ppo()

    # 使用Tune的训练
    print("\nTraining with Tune:")
    best_checkpoint = train_ppo_with_tune()

    # 加载最佳检查点
    best_algo = PPO(config=get_ppo_config(use_tune=False))
    best_algo.restore(best_checkpoint)

    # 比较评估
    print("\nEvaluating model trained without Tune:")
    # evaluate_ppo(algo)

    print("\nEvaluating model trained with Tune:")
    # evaluate_ppo(best_algo)

    ray.shutdown()