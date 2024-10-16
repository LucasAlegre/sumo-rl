"""Uses Ray's RLlib to view trained agents playing Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""

import argparse
import os

import ray
import supersuit as ss
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

from pettingzoo.butterfly import pistonball_v6


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


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

ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)


def env_creator():
    env = pistonball_v6.env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
        render_mode="rgb_array",
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 3)
    return env


env = env_creator()
env_name = "pistonball_v6"
register_env(env_name, lambda config: PettingZooEnv(env_creator()))

ray.init()

PPOagent = PPO.from_checkpoint(checkpoint_path)

reward_sum = 0
frame_list = []
i = 0
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    reward_sum += reward
    if termination or truncation:
        action = None
    else:
        action = PPOagent.compute_single_action(observation)

    env.step(action)
    i += 1
    if i % (len(env.possible_agents) + 1) == 0:
        img = Image.fromarray(env.render())
        frame_list.append(img)
env.close()

print(reward_sum)
frame_list[0].save(
    "out.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
)

"""
这个程序是用于加载预训练的 PPO (Proximal Policy Optimization) 模型，并在 Pistonball 环境中运行和可视化智能体的表现。让我详细分析其功能和逻辑：

1. 导入必要的库和模块。

2. 定义 CNNModelV2 类：
   - 这是一个自定义的卷积神经网络模型，用于处理游戏的视觉输入。
   - 包含卷积层、全连接层、策略函数和价值函数。

3. 设置命令行参数解析：
   - 允许用户指定预训练模型的检查点路径。

4. 注册自定义模型：
   - 使用 ModelCatalog.register_custom_model 注册 CNNModelV2。

5. 定义环境创建函数 env_creator：
   - 创建 Pistonball 环境并应用一系列包装器（如颜色减少、调整大小等）。

6. 初始化 Ray 和加载预训练模型：
   - 使用指定的检查点路径加载 PPO 模型。

7. 运行模拟：
   - 重置环境并开始迭代。
   - 对每个智能体：
     - 获取观察、奖励和终止信息。
     - 使用预训练模型计算动作。
     - 执行动作并更新环境。
   - 定期渲染环境状态并保存为图像。

8. 计算总奖励并保存可视化结果：
   - 累计每步的奖励。
   - 将渲染的帧保存为 GIF 动画。

主要功能：
1. 加载预训练的强化学习模型。
2. 在 Pistonball 环境中运行该模型。
3. 可视化智能体的行为。
4. 计算并输出总奖励。
5. 生成一个展示整个过程的 GIF 动画。

运行逻辑：
1. 解析命令行参数获取检查点路径。
2. 设置环境和模型。
3. 加载预训练模型。
4. 在环境中迭代运行模型，同时收集渲染帧。
5. 计算总奖励。
6. 将收集的帧保存为 GIF 文件。

这个程序展示了如何使用预训练的强化学习模型来展示其在特定环境中的表现，并提供了直观的可视化输出。它对于理解模型的行为、调试和展示训练结果非常有用。
"""

"""
ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

这行代码 `ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)` 在程序中起着关键的作用。让我详细解释它的功能和重要性：

1. 注册自定义模型：
   这行代码将自定义定义的 `CNNModelV2` 类注册到 Ray RLlib 的模型目录中。

2. 模型识别：
   通过注册，RLlib 能够识别并使用名为 "CNNModelV2" 的自定义模型。这个名称在加载预训练模型时用于识别正确的网络架构。

3. 一致性保证：
   确保加载预训练模型时使用的模型结构与训练时完全相同。这对于正确加载模型权重和执行推理至关重要。

4. 灵活性：
   允许在不修改 RLlib 核心代码的情况下使用自定义神经网络架构。

5. 错误预防：
   如果没有这行代码，在尝试加载使用这个自定义模型训练的检查点时，程序可能会失败或产生错误。

6. 与训练阶段的对应：
   这行代码需要与训练脚本中的相应代码匹配，确保播放脚本使用与训练时相同的模型定义。

7. 环境特定优化：
   允许使用为特定环境（在这里是 Pistonball）优化的自定义神经网络结构。

8. 可重用性：
   通过注册，这个自定义模型可以在程序的其他部分被引用和使用，增加了代码的模块化和可重用性。

总之，这行代码是连接自定义模型定义和 RLlib 框架的关键桥梁，确保了预训练模型可以被正确加载和使用。它体现了 RLlib 框架的灵活性，允许研究者和开发者使用自定义的神经网络架构来解决特定的强化学习问题。
"""