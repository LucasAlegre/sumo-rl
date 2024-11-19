import pprint
from pathlib import Path

from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
)

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

parser = add_rllib_example_script_args()
# Use `parser` to add your own custom command line options to this script
# and (if needed) use their values to set up `config` below.
args = parser.parse_args()

assert (
        args.env == "CartPole-v1" or args.env is None
), "This tuned example works only with `CartPole-v1`."

# Define the data paths.
data_path = "ray_rllib/data/cartpole/cartpole-v1_large"
base_path = Path(__file__).parents[1]
print(f"base_path={base_path}")
data_path = "local://" / base_path / data_path
print(f"data_path={data_path}")

# Define the BC config.
config = (
    BCConfig()
    .environment(env="CartPole-v1")
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .evaluation(
        evaluation_interval=3,
        evaluation_num_env_runners=1,
        evaluation_duration=5,
        evaluation_parallel_to_training=True,
    )
    # Note, the `input_` argument is the major argument for the
    # new offline API. Via the `input_read_method_kwargs` the
    # arguments for the `ray.data.Dataset` read method can be
    # configured. The read method needs at least as many blocks
    # as remote learners.
    .offline_data(
        input_=[data_path.as_posix()],
        # Define the number of reading blocks, these should be larger than 1
        # and aligned with the data size.
        input_read_method_kwargs={
            "override_num_blocks": max((args.num_env_runners or 1) * 2, 2)
        },
        # Concurrency defines the number of processes that run the
        # `map_batches` transformations. This should be aligned with the
        # 'prefetch_batches' argument in 'iter_batches_kwargs'.
        map_batches_kwargs={"concurrency": 2, "num_cpus": 2},
        # This data set is small so do not prefetch too many batches and use no
        # local shuffle.
        iter_batches_kwargs={
            "prefetch_batches": 1,
            "local_shuffle_buffer_size": None,
        },
        # The number of iterations to be run per learner when in multi-learner
        # mode in a single RLlib training iteration. Leave this to `None` to
        # run an entire epoch on the dataset during a single RLlib training
        # iteration. For single-learner mode, 1 is the only option.
        dataset_num_iters_per_learner=1 if not args.num_env_runners else None,
    )
    .training(
        train_batch_size_per_learner=1024,
        # To increase learning speed with multiple learners,
        # increase the learning rate correspondingly.
        lr=0.0008 * (args.num_env_runners or 1) ** 0.5,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
        ),
    )
)

stop = {
    f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": 350.0,
    TRAINING_ITERATION: 350,
}

if __name__ == "__main__":
    run_rllib_example_script_experiment(config, args, stop=stop)

"""
BC（Behavior Cloning，行为克隆）算法：

BC算法是一种模仿学习的方法，其主要目的是让AI直接模仿和复制已有的示范行为。从图中我们可以看到它的几个关键特点：

1. 数据处理流程：
- 首先从本地或云端读取示范数据（图中显示的是棒球相关的动作数据）
- 使用n个数据工作器（Data Workers）并行处理数据
- 这些数据会经过预处理器（OfflinePreLearner）转换成训练用的数据格式

2. 学习架构：
- 使用m个学习器（Learners）在GPU上进行并行学习
- 每个学习器包含完整的强化学习模块（RLModule）
- 通过损失函数（loss）和优化器（optim）来更新模型

3. 重要特征：
- BC算法基于RLlib库实现
- 它直接继承自MARWIL（Weighted Imitation Learning）算法
- 唯一的区别是beta参数设为0.0
- 这意味着BC只关注模仿行为本身，而不考虑行为带来的奖励

4. 训练过程：
```python
def training_step(...):
    data_iterator = OfflineData.sample()
    LearnerGroup.update(data_iterator)
```
这个简化的训练步骤显示了它如何从离线数据中采样，并用这些数据更新学习器组。

BC算法的主要目标是让AI准确复制示范数据中的行为策略，而不关心这些行为实际产生的奖励结果。这使得它特别适合于那些有明确正确行为示范，但难以定义明确奖励函数的场景。

这种方法的优点是实现简单直接，缺点是它可能会简单地复制示范者的行为，包括其中可能存在的次优决策。
"""


"""
这是一个基于CartPole环境的行为克隆BC (Behavior Cloning)实现：

1. 核心配置部分：

```
config = BCConfig()
    .environment(env="CartPole-v1")  # 使用CartPole-v1环境
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
```

2. 离线数据处理配置：
```
.offline_data(
    input_=[data_path.as_posix()],  # 指定离线数据路径
    input_read_method_kwargs={
        "override_num_blocks": max((args.num_env_runners or 1) * 2, 2)
    },
    map_batches_kwargs={"concurrency": 2, "num_cpus": 2},  # 并行处理设置
    iter_batches_kwargs={
        "prefetch_batches": 1,
        "local_shuffle_buffer_size": None,
    }
)
```

3. 训练参数配置：
```
.training(
    train_batch_size_per_learner=1024,  # 每个learner的批次大小
    # 学习率会随着环境运行器数量的增加而调整
    lr=0.0008 * (args.num_env_runners or 1) ** 0.5,
)
```

4. 神经网络模型配置：
```
.rl_module(
    model_config=DefaultModelConfig(
        fcnet_hiddens=[256, 256],  # 两层隐藏层，每层256个节点
    ),
)
```

5. 停止条件设置：
```
stop = {
    f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": 350.0,  # 平均回报达到350
    TRAINING_ITERATION: 350,  # 或训练迭代次数达到350
}
```

这段代码展示了BC算法的几个关键特点：

1. **数据并行处理**：
   - 使用多个环境运行器并行处理数据
   - 通过concurrency和num_cpus参数控制并行度

2. **批量学习**：
   - 设置了较大的批次大小(1024)来提高学习效率
   - 学习率会随着并行度的增加而自适应调整

3. **模型架构**：
   - 使用简单的前馈神经网络
   - 包含两个256节点的隐藏层

4. **评估机制**：
   - 每3次训练进行一次评估
   - 评估过程与训练并行
   - 设置了明确的性能目标（平均回报350）

5. **离线学习特点**：
   - 从预先收集的数据中学习（CartPole环境的示范数据）
   - 没有直接与环境交互
   - 通过预取机制（prefetch_batches）优化数据加载

这个实现展示了BC算法在实际应用中如何高效地从示范数据中学习策略，特别适合于像CartPole这样的相对简单的控制任务。代码通过并行处理和批量学习来提高训练效率，同时保持了实现的简洁性。
"""