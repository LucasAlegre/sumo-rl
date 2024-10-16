import argparse
import os
import sys
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from pprint import pprint

import logging

logging.basicConfig(level=logging.DEBUG)
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


def train_resco_ppo(env_name="arterial4x4", num_iterations=200, use_gpu=False, num_env_runners=4):
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

    # logger.debug("=====================results.trials[0].last_result:=====================")
    # pprint(results.trials[0].last_result)

    def get_episode_reward_mean(trial):
        return trial.last_result["env_runners"]["episode_reward_mean"]

    best_trial = results.get_best_trial("env_runners/episode_reward_mean", mode="max")
    logger.debug(f"最佳试验: {best_trial}")

    if best_trial:
        logger.debug("=====================最佳试验的详细信息=====================")
        logger.debug("最佳试验的配置:")
        pprint(best_trial, indent=2)
        logger.debug(f"最佳试验的训练时长: {best_trial.last_result['time_total_s']} 秒")
        logger.debug(f"最佳试验完成的迭代次数: {best_trial.last_result['training_iteration']}")

    best_reward = get_episode_reward_mean(best_trial)
    logger.debug(f"最佳试验的平均奖励: {best_reward}")

    if best_trial:
        best_checkpoint = results.get_best_checkpoint(best_trial, "env_runners/episode_reward_mean", mode="max")
        logger.debug(f"最佳检查点: {best_checkpoint}")
    else:
        logger.debug("未找到最佳试验，无法获取检查点。")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练PPO-RESCO代理")
    parser.add_argument("--env_name", type=str, default="arterial4x4", help="环境名称")
    parser.add_argument("--num_iterations", type=int, default=200, help="训练迭代次数")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU")
    parser.add_argument("--num_env_runners", type=int, default=4, help="环境运行器数量")

    args = parser.parse_args()

    train_resco_ppo(
        env_name=args.env_name,
        num_iterations=args.num_iterations,
        use_gpu=args.use_gpu,
        num_env_runners=args.num_env_runners
    )

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
