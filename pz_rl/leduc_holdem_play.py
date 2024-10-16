"""Uses Ray's RLlib to view trained agents playing Leduoc Holdem.

Author: Rohan (https://github.com/Rohan138)
"""

import argparse
import os

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from leduc_holdem_train import TorchMaskedActions

from pettingzoo.classic import leduc_holdem_v4

os.environ["SDL_VIDEODRIVER"] = "dummy"

parser = argparse.ArgumentParser(
    description="Render pretrained policy loaded from checkpoint"
)
parser.add_argument(
    "--checkpoint-path",
    help="Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`",
)

args = parser.parse_args()


if args.checkpoint_path is None:
    print("The following arguments are required: --checkpoint-path")
    exit(0)

checkpoint_path = os.path.expanduser(args.checkpoint_path)


alg_name = "DQN"
ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
# function that outputs the environment you wish to register.


def env_creator():
    env = leduc_holdem_v4.env()
    return env


env = env_creator()
env_name = "leduc_holdem_v4"
register_env(env_name, lambda config: PettingZooEnv(env_creator()))


ray.init()
DQNAgent = Algorithm.from_checkpoint(checkpoint_path)

reward_sums = {a: 0 for a in env.possible_agents}
i = 0
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    obs = observation["observation"]
    reward_sums[agent] += reward
    if termination or truncation:
        action = None
    else:
        print(DQNAgent.get_policy(agent))
        policy = DQNAgent.get_policy(agent)
        batch_obs = {
            "obs": {
                "observation": np.expand_dims(observation["observation"], 0),
                "action_mask": np.expand_dims(observation["action_mask"], 0),
            }
        }
        batched_action, state_out, info = policy.compute_actions_from_input_dict(
            batch_obs
        )
        single_action = batched_action[0]
        action = single_action

    env.step(action)
    i += 1
    env.render()

print("rewards:")
print(reward_sums)

"""
这个程序是用于加载预训练的 DQN 模型并在 Leduc Hold'em 扑克游戏环境中运行和可视化智能体的表现。让我详细分析其功能和逻辑：

1. 导入必要的库和模块：
   包括 Ray、RLlib、PettingZoo 等，以及自定义的 TorchMaskedActions 模型。

2. 设置命令行参数解析：
   允许用户指定预训练模型的检查点路径。

3. 注册自定义模型和环境：
   - 使用 ModelCatalog.register_custom_model 注册 TorchMaskedActions 模型。
   - 定义并注册 Leduc Hold'em 环境。

4. 初始化 Ray 和加载预训练模型：
   使用指定的检查点路径加载 DQN 模型。

5. 运行模拟：
   - 初始化环境和奖励累计字典。
   - 进入主循环，遍历环境中的每个智能体：
     a. 获取观察、奖励和终止信息。
     b. 累计奖励。
     c. 如果游戏未结束，使用加载的模型计算动作：
        - 获取对应智能体的策略。
        - 准备观察数据批次。
        - 使用策略计算动作。
     d. 执行动作并更新环境。
     e. 渲染环境状态。

6. 输出结果：
   打印每个智能体的累计奖励。

主要功能：
1. 加载预训练的 DQN 模型。
2. 在 Leduc Hold'em 环境中运行该模型。
3. 可视化智能体的行为。
4. 计算并输出每个智能体的总奖励。

关键点：
1. 使用 PettingZooEnv 包装器来适配 RLlib 和 PettingZoo 环境。
2. 处理多智能体环境，为每个智能体单独计算动作和累计奖励。
3. 使用预训练模型的策略来计算动作，而不是随机选择。
4. 支持渲染环境状态，便于观察游戏进程。

这个程序展示了如何使用预训练的强化学习模型来展示其在复杂的多智能体扑克游戏环境中的表现。它对于理解模型的行为、调试和展示训练结果非常有用。通过可视化和奖励统计，可以直观地评估模型的性能和策略。
"""