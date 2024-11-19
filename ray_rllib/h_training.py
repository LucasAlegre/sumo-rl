# @OldAPIStack

"""Example of hierarchical training using the multi-agent API.

The example env is that of a "windy maze". The agent observes the current wind
direction and can either choose to stand still, or move in that direction.

You can try out the env directly with:

    $ python hierarchical_training.py --flat

A simple hierarchical formulation involves a high-level agent that issues goals
(i.e., go north / south / east / west), and a low-level agent that executes
these goals over a number of time-steps. This can be implemented as a
multi-agent environment with a top-level agent and low-level agents spawned
for each higher-level action. The lower level agent is rewarded for moving
in the right direction.

You can try this formulation with:

    $ python hierarchical_training.py  # gets ~100 rew after ~100k timesteps

Note that the hierarchical formulation actually converges slightly slower than
using --flat in this example.
"""

import argparse
from gymnasium.spaces import Discrete, Tuple
import logging
import os

import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.envs.classes.windy_maze_env import (
    WindyMazeEnv,
    HierarchicalWindyMazeEnv,
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument("--flat", action="store_true")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(local_mode=args.local_mode)

    stop = {
        TRAINING_ITERATION: args.stop_iters,
        NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    }

    if args.flat:
        results = tune.Tuner(
            "PPO",
            run_config=air.RunConfig(stop=stop),
            param_space=(
                PPOConfig()
                .api_stack(
                    enable_env_runner_and_connector_v2=False,
                    enable_rl_module_and_learner=False,
                )
                .environment(WindyMazeEnv)
                .env_runners(num_env_runners=0)
                .framework(args.framework)
            ).to_dict(),
        ).fit()
    else:
        maze = WindyMazeEnv(None)

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id.startswith("low_level_"):
                return "low_level_policy"
            else:
                return "high_level_policy"

        config = (
            PPOConfig()
            .api_stack(
                enable_env_runner_and_connector_v2=False,
                enable_rl_module_and_learner=False,
            )
            .environment(HierarchicalWindyMazeEnv)
            .framework(args.framework)
            .env_runners(num_env_runners=0)
            .training(entropy_coeff=0.01)
            .multi_agent(
                policies={
                    "high_level_policy": (
                        None,
                        maze.observation_space,
                        Discrete(4),
                        PPOConfig.overrides(gamma=0.9),
                    ),
                    "low_level_policy": (
                        None,
                        Tuple([maze.observation_space, Discrete(4)]),
                        maze.action_space,
                        PPOConfig.overrides(gamma=0.0),
                    ),
                },
                policy_mapping_fn=policy_mapping_fn,
            )
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        )

        results = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop, verbose=1),
        ).fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()

"""
# 程序功能概述

这是一个演示分层强化学习的示例程序，主要实现了一个"风向迷宫"(Windy Maze)环境的训练。程序提供了两种训练模式：
1. 扁平(Flat)模式：单个智能体直接学习动作
2. 分层(Hierarchical)模式：通过高层和低层智能体协作来学习

## 环境设计

### 风向迷宫(Windy Maze)环境
- 智能体可以观察当前的风向
- 可选动作:
  - 站立不动
  - 随风向移动

## 分层学习架构

### 高层智能体(High-level Agent)
- 功能：决定目标方向(北/南/东/西)
- 观察空间：与基础环境相同
- 动作空间：4个离散动作(四个方向)
- 特点：
  - 使用较长期的奖励(gamma=0.9)
  - 负责战略决策

### 低层智能体(Low-level Agent)
- 功能：执行高层智能体下达的指令
- 观察空间：基础环境状态 + 高层指令
- 动作空间：与基础环境相同
- 特点：
  - 使用即时奖励(gamma=0.0)
  - 专注于战术执行

## 关键实现细节

1. 策略映射
```
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id.startswith("low_level_"):
        return "low_level_policy"
    else:
        return "high_level_policy"
```

2. 训练配置
- 使用 PPO (Proximal Policy Optimization) 算法
- 多智体配置：
  - 高层策略：gamma=0.9 的长期规划
  - 低层策略：gamma=0.0 的即时执行
- 熵系数：0.01，用于鼓励探索

## 训练参数
- 最大迭代次数：200
- 最大时间步数：100,000
- 支持 TensorFlow 和 PyTorch 框架
- 可配置 GPU 使用

## 性能说明
- 在约10万时间步后可达到约100的奖励
- 有趣的是，分层方法在这个简单环境中收敛速度实际略慢于扁平方法
"""
