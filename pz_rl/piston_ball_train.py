"""Uses Ray's RLlib to train agents to play Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
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


def env_creator(args):
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    ray.init()

    local_dir = os.path.abspath("./ray_results")
    env_name = "pistonball_v6"
    storage_path = os.path.join(local_dir, env_name)

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .env_runners(num_env_runners=4, rollout_fragment_length=128)
        .training(
            model={"custom_model": "CNNModelV2"}, # 添加这一行
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=storage_path,
        config=config.to_dict(),
    )
    
"""
这个程序是一个使用 Ray 的 RLlib 框架来训练智能体玩 Pistonball 游戏的实现。让我详细分析其功能和运行逻辑：

1. 导入必要的库和模块：
   - Ray 和 RLlib 用于分布式强化学习
   - SuperSuit 用于环境预处理
   - PettingZoo 提供 Pistonball 环境

2. 定义自定义 CNN 模型 (CNNModelV2):
   - 继承自 TorchModelV2 和 nn.Module
   - 实现了一个卷积神经网络，用于处理图像输入
   - 包含策略函数和价值函数

3. 定义环境创建函数 (env_creator):
   - 创建 Pistonball 并行环境，设置各种参数
   - 应用多个环境包装器（颜色减少、数据类型转换、调整大小、归一化和帧堆叠）

4. 主函数逻辑:
   a. 初始化 Ray
   b. 设置结果存储路径
   c. 注册环境和自定义模型
   d. 配置 PPO 算法:
      - 设置环境参数
      - 配置训练参数（学习率、批量大小等）
      - 设置框架为 PyTorch
      - 配置 GPU 使用
   e. 运行训练:
      - 使用 tune.run 启动训练过程
      - 设置停止条件、检查点频率等

运行逻辑：
1. 程序启动，初始化 Ray
2. 创建并注册 Pistonball 环境
3. 注册自定义 CNN 模型
4. 配置 PPO 算法参数
5. 开始训练循环：
   - 并行收集环境交互数据
   - 使用收集的数据更新策略网络
   - 定期保存检查点
   - 达到指定的时间步数后停止训练

主要功能：
- 使用深度强化学习（PPO 算法）训练智能体在 Pistonball 游戏中表现
- 利用自定义 CNN 模型处理游戏的视觉输入
- 通过 Ray 实现分布式训练，提高效率
- 使用 SuperSuit 对环境进行预处理，标准化输入
- 支持 GPU 训练（如果可用）
- 提供灵活的配置选项，如训练步数、批量大小等
- 自动保存训练检查点，便于后续分析或继续训练

这个程序展示了如何使用现代强化学习框架来训练复杂环境中的智能体，结合了深度学习和强化学习技术，并利用了分布式计算来提高训练效率。
"""

"""
register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))



这行代码是在 Ray RLlib 中注册一个自定义环境。让我详细解释一下这行代码的作用和组成部分：

1. `register_env` 函数：
   这是 Ray 提供的一个函数，用于注册自定义环境。它允许你给环境一个名字，并提供一个创建环境实例的函数。

2. `env_name`：
   这是你给环境起的名字，在之前的代码中定义为 "pistonball_v6"。这个名字后续会被用来引用这个环境。

3. `lambda config: ...`：
   这是一个匿名函数（lambda 函数），它接受一个 `config` 参数。这个函数定义了如何创建环境实例。

4. `ParallelPettingZooEnv`：
   这是 RLlib 提供的一个包装器，用于将 PettingZoo 的并行环境转换为 RLlib 可以使用的格式。

5. `env_creator(config)`：
   这调用了之前定义的 `env_creator` 函数，该函数创建并配置了 Pistonball 环境。

整体功能：
- 这行代码注册了一个名为 "pistonball_v6" 的环境。
- 当 RLlib 需要创建这个环境的实例时，它会调用这个 lambda 函数。
- lambda 函数首先使用 `env_creator` 创建一个 Pistonball 环境，然后用 `ParallelPettingZooEnv` 包装它。
- 这样做使得 RLlib 能够与 PettingZoo 的 Pistonball 环境无缝集成。

为什么这样做：
1. 环境注册：允许 RLlib 知道如何创建和使用这个自定义环境。
2. 灵活性：通过使用 lambda 函数，可以在创建环境时传入额外的配置。
3. 兼容性：`ParallelPettingZooEnv` 确保 PettingZoo 环境可以在 RLlib 中正常工作。
4. 封装：将环境创建的复杂性隐藏在一个简单的接口后面。

这种方式使得在 RLlib 中使用自定义环境变得简单和标准化，同时保持了足够的灵活性来处理各种不同类型的环境。

"""