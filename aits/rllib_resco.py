import argparse
import os
import sys
import ray
import torch
from ray import tune
from ray.rllib.algorithms import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from pprint import pprint
import re
import logging

os.environ['COLORFUL_DISABLE'] = '1'

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sys.path.append('../')
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mysumo.envs.resco_envs import grid4x4, arterial4x4, cologne1, cologne3, cologne8, ingolstadt1, ingolstadt7, ingolstadt21

def env_creator(config):
    env_name = config.get("env_name", "arterial4x4")
    use_gui = config.get("use_gui", False)
    yellow_time = config.get("yellow_time", 2)
    fixed_ts = config.get("fixed_ts", False)

    env_mapping = {
        "arterial4x4": arterial4x4,
        "grid4x4": grid4x4,
        "cologne1": cologne1,
        "cologne3": cologne3,
        "cologne8": cologne8,
        "ingolstadt1": ingolstadt1,
        "ingolstadt7": ingolstadt7,
        "ingolstadt21": ingolstadt21
    }

    if env_name not in env_mapping:
        raise ValueError(f"未知的环境名称: {env_name}")

    env_func = env_mapping[env_name]
    out_csv_name = f"outputs/{env_name}/{env_name}"

    env = env_func(out_csv_name=out_csv_name, use_gui=use_gui, yellow_time=yellow_time, fixed_ts=fixed_ts)
    return env


register_env("sumo_env", lambda config: ParallelPettingZooEnv(env_creator(config)))


def train_resco_ppo(env_name="arterial4x4", num_iterations=200, use_gpu=False, num_env_runners=4, checkpoint_path=None):
    logger.debug("=====================train_resco_ppo=====================")
    ray.init()

    config = (
        PPOConfig()
        .environment("sumo_env", env_config={
            "env_name": env_name,
            "use_gui": False,
            "yellow_time": 2,
            "fixed_ts": False
        })
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=128
        )
        .training(
            train_batch_size=1024,
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
        "time_total_s": 3600,  # 1小时
    }

    # 使用绝对路径
    storage_path = os.path.abspath("./ray_results")

    logger.debug("=====================start tune.run====================")

    # 如果提供了检查点路径，则从检查点恢复训练
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"从检查点恢复训练: {checkpoint_path}")
        results = tune.run(
            "PPO",
            config=config.to_dict(),
            stop=stop,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            name="resco_ppo",
            storage_path=storage_path,
            verbose=3,
            log_to_file=True,
            raise_on_failed_trial=False,
            restore=checkpoint_path,  # 从检查点恢复
        )
    else:
        print("从头开始训练")
        results = tune.run(
            "PPO",
            config=config.to_dict(),
            stop=stop,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            name="resco_ppo",
            storage_path=storage_path,
            verbose=3,
            log_to_file=True,
            raise_on_failed_trial=False,
        )

    logger.debug("=====================end tune.run=====================")

    best_trial = results.get_best_trial("env_runners/episode_reward_mean", mode="max")

    if best_trial:
        print("=====================最佳试验的详细信息=====================")
        print(f"最佳试验的训练时长: {best_trial.last_result['time_total_s']} 秒")
        print(f"最佳试验完成的迭代次数: {best_trial.last_result['training_iteration']}")
        print(f"最佳试验的平均奖励: {best_trial.last_result['env_runners']['episode_reward_mean']}")
        best_checkpoint = results.get_best_checkpoint(best_trial, "env_runners/episode_reward_mean", mode="max")
        print(f"最佳检查点: {best_checkpoint}")
        loaded_model = PPO.from_checkpoint(best_checkpoint)
        save_dir = os.path.join(storage_path, "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best_model_{env_name}")
        loaded_model.save(save_path)
        print(f"最佳模型已保存到: {save_path}")
    else:
        print("=====================未找到最佳试验，无法获取检查点和保存模型=====================")

    ray.shutdown()

# checkpoint_path="/Users/xnpeng/sumoptis/sumo-rl/ray_results/resco_ppo/PPO_sumo_env_81136_00000_0_2024-10-16_18-43-39/checkpoint_000000"
# saved_model_path="/Users/xnpeng/sumoptis/sumo-rl/ray_results/saved_models/best_model_arterial4x4"
def predict_resco_ppo(saved_model_path: str, env_name="arterial4x4", use_gui=False):
    logger.debug("=====================predict_resco_ppo=====================")

    # 检查 Ray 是否已初始化，如果没有，则初始化
    if not ray.is_initialized():
        ray.init()

    # 加载保存的模型
    loaded_model = PPO.from_checkpoint(saved_model_path)
    env = env_creator({"env_name": env_name, "use_gui": use_gui})  # 使用GUI进行可视化

    obs, _ = env.reset()

    done = False
    total_reward = 0
    episode_steps = 0
    max_steps = 1000  # 设置最大步数，防止无限循环

    while not done and episode_steps < max_steps:
        actions = {}
        for agent_id, agent_obs in obs.items():
            # 将 NumPy 数组转换为 PyTorch 张量
            agent_obs_tensor = torch.FloatTensor(agent_obs)
            actions[agent_id] = loaded_model.compute_single_action(agent_obs_tensor)

        obs, rewards, dones, _, infos = env.step(actions)  # 注意这里的返回值变化
        total_reward += sum(rewards.values())
        done = all(dones.values())  # 当所有智能体都完成时，整个环境才算完成
        episode_steps += 1

    print(f"总奖励: {total_reward}")
    print(f"总步数: {episode_steps}")

    # 如果在这个函数中初始化了 Ray，记得在结束时关闭
    if ray.is_initialized():
        ray.shutdown()


def evaluate_resco_ppo(saved_model_path: str, env_name="arterial4x4", use_gpu=False):
    logger.debug("=====================evaluate_resco_ppo=====================")

    # 检查 Ray 是否已初始化，如果没有，则初始化
    if not ray.is_initialized():
        ray.init()

    # 加载保存的模型
    loaded_model = PPO.from_checkpoint(saved_model_path)
    loaded_model.evaluate()
    env = env_creator({"env_name": env_name, "use_gui": use_gpu})  # 使用GUI进行可视化

    obs, _ = env.reset()

    # 评估模型性能
    num_episodes = 5
    all_rewards = []
    all_steps = []
    max_steps = 1000  # 设置最大步数，防止无限循环

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < max_steps:
            actions = {}
            for agent_id, agent_obs in obs.items():
                # 将 NumPy 数组转换为 PyTorch 张量
                agent_obs_tensor = torch.FloatTensor(agent_obs)
                actions[agent_id] = loaded_model.compute_single_action(agent_obs_tensor)

            obs, rewards, dones, _, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            done = all(dones.values())
            episode_steps += 1

        all_rewards.append(episode_reward)
        all_steps.append(episode_steps)

    print(f"平均奖励: {sum(all_rewards) / num_episodes}")
    print(f"平均步数: {sum(all_steps) / num_episodes}")

    # 如果在这个函数中初始化了 Ray，记得在结束时关闭
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练PPO-RESCO代理")
    parser.add_argument("--env_name", type=str, default="arterial4x4", help="环境名称")
    parser.add_argument("--num_iterations", type=int, default=200, help="训练迭代次数")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU")
    parser.add_argument("--use_gui", action="store_true", help="是否使用GUI进行可视化")
    parser.add_argument("--num_env_runners", type=int, default=4, help="环境运行器数量")
    parser.add_argument("--operation", type=str, default="TRAIN", help="操作指示")
    parser.add_argument("--saved_model_path", type=str, help="模型路径")
    parser.add_argument("--checkpoint_path", type=str, help="检查点路径")
    
    args = parser.parse_args()

    if args.operation == "TRAIN":
        train_resco_ppo(
            env_name=args.env_name,
            num_iterations=args.num_iterations,
            use_gpu=args.use_gpu,
            num_env_runners=args.num_env_runners,
            checkpoint_path=args.checkpoint_path
        )
    elif args.operation == "PREDICT":
        predict_resco_ppo(args.saved_model_path,
                          env_name=args.env_name,
                          use_gui=args.use_gui)
    elif args.operation == "EVALUATE":
        evaluate_resco_ppo(args.saved_model_path,
                          env_name=args.env_name,
                          use_gpu=args.use_gpu)
    else:
        raise Exception("no such operation")

"""
这个程序是一个使用Ray RLlib框架来训练强化学习代理的示例，主要用于优化交通信号控制。分析一下程序的设计原理和预期结果：

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

"""
根据您提供的代码和描述，我可以为您分析如何评估训练结果以及如何使用训练后的模型进行推理：

1. 评估训练结果：

a) 平均奖励：
   程序使用 `episode_reward_mean` 作为主要的评估指标。这在代码中体现为：
   ```python
   def get_episode_reward_mean(trial):
       return trial.last_result["env_runners"]["episode_reward_mean"]
   
   best_trial = results.get_best_trial("env_runners/episode_reward_mean", mode="max")
   best_reward = get_episode_reward_mean(best_trial)
   logger.debug(f"最佳试验的平均奖励: {best_reward}")
   ```

b) 训练时长和迭代次数：
   程序记录了最佳试验的训练时长和完成的迭代次数：
   ```python
   logger.debug(f"最佳试验的训练时长: {best_trial.last_result['time_total_s']} 秒")
   logger.debug(f"最佳试验完成的迭代次数: {best_trial.last_result['training_iteration']}")
   ```

c) 检查点：
   程序保存了性能最佳的模型检查点：
   ```python
   best_checkpoint = results.get_best_checkpoint(best_trial, "env_runners/episode_reward_mean", mode="max")
   logger.debug(f"最佳检查点: {best_checkpoint}")
   ```

d) 日志和输出：
   训练过程中的详细日志和输出数据被保存在 "outputs/arterial4x4/ppo" 目录下。

2. 使用训练后的模型进行推理：

虽然当前代码没有直接包含推理部分，但您可以按照以下步骤使用训练后的模型进行推理：

a) 加载训练好的模型：
   ```python
   from ray.rllib.algorithms.ppo import PPO
   
   # 加载最佳检查点
   loaded_model = PPO.from_checkpoint(best_checkpoint)
   ```

b) 创建环境实例：
   ```python
   env = env_creator({"env_name": "arterial4x4", "use_gui": True})  # 使用GUI进行可视化
   ```

c) 运行推理：
   ```python
   obs = env.reset()
   done = False
   total_reward = 0
   
   while not done:
       action = loaded_model.compute_single_action(obs)
       obs, reward, done, info = env.step(action)
       total_reward += reward
       env.render()  # 如果需要可视化
   
   print(f"Total reward: {total_reward}")
   ```

d) 评估指标：
   您可以在推理过程中收集更多的指标，例如：
   - 平均等待时间
   - 通过交叉路口的车辆数量
   - 交通流量
   - 燃料消耗或排放

e) 多次运行：
   为了获得更可靠的结果，您应该多次运行推理过程，并计算平均性能。

f) 比较基准：
   将训练后的模型与固定时间信号控制或其他传统方法进行比较，以评估改进程度。

g) 不同场景测试：
   在不同的交通流量条件下测试模型，评估其适应性。

要实现这些推理和评估步骤，您需要在现有代码基础上添加新的函数或脚本。这可能包括创建一个单独的推理脚本，该脚本加载训练好的模型，运行多次模拟，并收集详细的性能指标。

此外，考虑使用可视化工具（如 SUMO-GUI）来直观地观察模型的表现，这对于理解和验证模型的行为非常有帮助。

"""

"""
关于从检查点恢复训练。
--num_iterations必须大于之前的数字，比如原来是10轮，现在必须大于10轮，比如num_iterations=20，即从第11轮开始训练。
如果num_iterations=10，则认为训练已经完成了。
"""

"""
MacBook 
最佳试验的训练时长: 869.3429336547852 秒
最佳试验完成的迭代次数: 50
最佳试验的平均奖励: -27.194285714285744
最佳检查点: Checkpoint(filesystem=local, path=/Users/xnpeng/sumoptis/sumo-rl/ray_results/resco_ppo/PPO_sumo_env_e6e52_00000_0_2024-10-17_17-19-25/checkpoint_000000)
最佳模型已保存到: /Users/xnpeng/sumoptis/sumo-rl/ray_results/saved_models/best_model_arterial4x4
"""
"""
Ubuntu GPU
最佳试验的训练时长: 1647.0169277191162 秒
最佳试验完成的迭代次数: 300
最佳试验的平均奖励: -9.396799999999988
最佳检查点: Checkpoint(filesystem=local, path=/home/kemove/Projects/sumo-rl/ray_results/resco_ppo/PPO_sumo_env_f0c0e_00000_0_2024-10-18_10-44-48/checkpoint_000016)

最佳试验的训练时长: 1045.5579164028168 秒
最佳试验完成的迭代次数: 400
最佳试验的平均奖励: -13.772099999999956
最佳检查点: Checkpoint(filesystem=local, path=/home/kemove/Projects/sumo-rl/ray_results/resco_ppo/PPO_sumo_env_78bf4_00000_0_2024-10-18_11-45-52/checkpoint_000000)

"""
