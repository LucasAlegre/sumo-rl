import os
import sys
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import arterial4x4


def env_creator(config):
    env = arterial4x4(out_csv_name="outputs/arterial4x4/ppo", use_gui=False, yellow_time=2, fixed_ts=False)
    return env


register_env("arterial4x4", lambda config: ParallelPettingZooEnv(env_creator(config)))


def train_ppo_resco(num_iterations=200, use_gpu=False, num_env_runners=4):
    ray.init()

    config = (
        PPOConfig()
        .environment("arterial4x4")
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=128
        )
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=None,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .framework("torch")
        .resources(num_gpus=int(use_gpu))
    )

    stop = {
        "training_iteration": num_iterations,
    }

   # 使用绝对路径
    storage_path = os.path.abspath("./ray_results")
    
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        name="ppo_arterial4x4",
        storage_path=storage_path,
    )

    best_checkpoint = results.get_best_checkpoint(results.get_best_trial("episode_reward_mean"), "episode_reward_mean")
    print(f"Best checkpoint: {best_checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    train_ppo_resco()

"""
这个程序是一个使用Ray RLlib框架来训练强化学习代理的示例，主要用于优化交通信号控制。让我为您分析一下程序的设计原理和预期结果：

1. 设计原理：

a) 环境设置：
- 程序使用SUMO（Simulation of Urban MObility）作为交通模拟器。
- 创建了一个4x4的十字路口网络环境（arterial4x4）。

b) 强化学习框架：
- 使用Ray RLlib作为强化学习框架。
- 采用PPO（Proximal Policy Optimization）算法进行训练。

c) 环境封装：
- 使用ParallelPettingZooEnv将自定义环境封装成RLlib可用的格式。

d) 训练配置：
- 设置了PPO算法的各种超参数，如学习率、批量大小等。
- 可以选择是否使用GPU进行训练。
- 支持多个环境并行运行（num_env_runners）。

e) 训练过程：
- 使用Ray Tune进行训练，支持分布式训练和超参数调优。
- 定期保存检查点，并在训练结束时保存最佳检查点。

2. 预期结果：

a) 性能提升：
- 随着训练的进行，期望代理能够学习到更好的交通信号控制策略。
- 预期会看到平均奖励（episode_reward_mean）随时间增加。

b) 最佳模型：
- 训练结束后，程序会输出性能最佳的检查点路径。

c) 输出数据：
- 训练过程中的数据会被保存在"outputs/arterial4x4/ppo"目录下。

d) 可视化：
- 虽然训练过程中没有使用GUI，但训练好的模型可以在有GUI的环境中进行测试和可视化。

e) 适应性：
- 训练出的模型应该能够适应不同的交通流量情况，比固定时间的信号控制更加灵活。

总的来说，这个程序旨在通过强化学习来优化交通信号控制，以提高交通效率，减少等待时间和拥堵。成功训练后，该模型应能根据实时交通状况动态调整信号灯时间，从而实现更智能的交通管理。

"""