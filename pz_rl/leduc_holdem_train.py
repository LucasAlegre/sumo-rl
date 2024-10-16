"""Uses Ray's RLlib to train agents to play Leduc Holdem.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
from gymnasium.spaces import Box, Discrete
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env

from pettingzoo.classic import leduc_holdem_v4

torch, nn = try_import_torch()


class TorchMaskedActions(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
            self,
            obs_space: Box,
            action_space: Discrete,
            num_outputs,
            model_config,
            name,
            **kw,
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = Box(
            shape=(obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len]
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


if __name__ == "__main__":
    ray.init()

    alg_name = "DQN"
    ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)


    # function that outputs the environment you wish to register.

    def env_creator():
        env = leduc_holdem_v4.env()
        return env


    env_name = "leduc_holdem_v4"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    local_dir = os.path.abspath("./ray_results")
    storage_path = os.path.join(local_dir, env_name)

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
            model={"custom_model": "pa_model"},
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space["player_0"], act_space["player_0"], {}),
                "player_1": (None, obs_space["player_1"], act_space["player_1"], {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="ERROR"
        )  # TODO: change to ERROR to match pistonball example
        .framework(framework="torch")
        .exploration(
            exploration_config={
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
            }
        )
    )

    tune.run(
        alg_name,
        name="DQN",
        stop={"timesteps_total": 10000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        config=config.to_dict(),
        storage_path=storage_path,
    )
   
"""
这个程序是使用 Ray 的 RLlib 框架来训练智能体玩 Leduc Hold'em 扑克游戏的实现。让我详细分析其功能和逻辑：

1. 导入必要的库和模块：
   包括 Ray、RLlib、PettingZoo 等。

2. 定义自定义模型 TorchMaskedActions：
   - 继承自 DQNTorchModel
   - 实现了动作掩码机制，用于处理非法动作
   - 包含一个嵌入模型来处理观察空间

3. 主函数逻辑：
   a. 初始化 Ray
   b. 注册自定义模型
   c. 定义环境创建函数
   d. 注册 Leduc Hold'em 环境
   e. 配置 DQN 算法：
      - 设置环境参数
      - 配置训练参数（批量大小、隐藏层等）
      - 设置多智能体策略
      - 配置 GPU 使用
      - 设置探索策略（Epsilon Greedy）
   f. 运行训练：
      - 使用 tune.run 启动训练过程
      - 设置停止条件、检查点频率等

主要功能：
1. 实现了一个适用于 Leduc Hold'em 的强化学习模型
2. 使用动作掩码处理非法动作
3. 支持多智能体训练（两个玩家）
4. 使用 DQN 算法进行训练
5. 实现了自定义的探索策略

关键点：
1. 自定义模型（TorchMaskedActions）处理了动作掩码，这对于扑克游戏很重要，因为并非所有动作在任何时候都是合法的。
2. 使用多智能体配置，为每个玩家定义了单独的策略。
3. 使用 Epsilon Greedy 探索策略，随时间逐渐减少随机探索。
4. 训练过程中会定期保存检查点。
5. 训练时间根据是否在 CI 环境中运行而有所不同（1000万或5万时间步）。

这个程序展示了如何使用先进的强化学习技术来训练智能体玩复杂的扑克游戏，处理了多智能体、部分可观察性和非法动作等挑战。它结合了深度学习（通过自定义神经网络模型）和强化学习（使用 DQN 算法）的技术。
"""