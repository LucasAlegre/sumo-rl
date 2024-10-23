from pprint import pprint

import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray import tune

# 初始化Ray
ray.init()

# 配置PPO算法训练参数
config = (
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

# 创建训练器
algo = config.build()

# 训练循环
for i in range(20):
    result = algo.train()

    # pprint(result)

    # 打印训练信息
    print(f"第 {i} 轮训练")
    metrics = result["env_runners"]
    print(f"episode_reward_mean: {metrics['episode_reward_mean']}")
    print(f"episode_len_mean: {metrics['episode_len_mean']}")

    # 每5轮评估一次
    if i % 5 == 0:
        eval_results = algo.evaluate()
        # pprint(eval_results)
        print(f"评估结果: {eval_results['env_runners']['episode_reward_mean']}")

# 保存训练好的模型
checkpoint_dir = algo.save()
print(f"模型保存在:")
pprint(checkpoint_dir.checkpoint.path)

# 测试训练好的智能体
env = gym.make("CartPole-v1", render_mode="human")
episode_reward = 0
obs, _ = env.reset()

while True:
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward

    if terminated or truncated:
        print(f"回合奖励: {episode_reward}")
        break

# 清理
algo.stop()
ray.shutdown()