import os
import tempfile
from time import sleep

import numpy as np
import torch
import torch.nn as nn
from ray.air import Result
from torch.optim import Adam

import ray.train.torch
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer


def train_func(config):
    n = 100
    # create a toy dataset
    # data   : X - dim = (n, 4)
    # target : Y - dim = (n, 1)
    X = torch.Tensor(np.random.normal(0, 1, size=(n, 4)))
    Y = torch.Tensor(np.random.uniform(0, 1, size=(n, 1)))
    # toy neural network : 1-layer
    model = nn.Linear(4, 1)
    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    # Wrap the model in DDP and move it to GPU.
    model = ray.train.torch.prepare_model(model)

    """
    创建了一个简单的线性回归模型，输入维度为4，输出维度为1。
    使用 Adam 优化器 和 均方误差损失。
    调用 ray.train.torch.prepare_model(model) 以使模型可以在 Ray 的分布式训练环境下运行。
    """

    # ====== Resume training state from the checkpoint. ======
    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state_dict = torch.load(
                os.path.join(checkpoint_dir, "model.pt"),
                weights_only=True
                # map_location=...,  # Load onto a different device if needed.
            )
            model.module.load_state_dict(model_state_dict)
            optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), weights_only=True)
            )
            start_epoch = (
                    torch.load(os.path.join(checkpoint_dir, "extra_state.pt"), weights_only=True)["epoch"] + 1
            )
    """
    如果有保存的检查点（checkpoint），则恢复模型的状态，包括：
    模型的参数（model.pt）。
    优化器的状态（optimizer.pt）。
    当前的训练epoch（extra_state.pt）。
    """
    # ========================================================

    for epoch in range(start_epoch, config["num_epochs"]):
        y = model.forward(X)
        loss = criterion(y, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = {"loss": loss.item()}

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0
            # In standard DDP training, where the model is the same across all ranks,
            # only the global rank 0 worker needs to save and report the checkpoint
            if train.get_context().get_world_rank() == 0 and should_checkpoint:
                # === Make sure to save all state needed for resuming training ===
                torch.save(
                    model.module.state_dict(),  # NOTE: Unwrap the model.
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(temp_checkpoint_dir, "optimizer.pt"),
                )
                torch.save(
                    {"epoch": epoch},
                    os.path.join(temp_checkpoint_dir, "extra_state.pt"),
                )
                # ================================================================
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report(metrics, checkpoint=checkpoint)

        # if epoch == 1:
        #     raise RuntimeError("故意出错。Intentional error to showcase restoration!")


"""
训练过程中：
每个epoch计算一次损失，并通过反向传播更新模型。
如果满足保存检查点的条件（例如每个epoch保存一次），则将当前模型、优化器和epoch状态保存到临时目录，并通过 train.report() 向 Ray 提交当前的损失和检查点。
如果出现错误（例如 epoch == 1），则展示训练恢复的功能。
"""

"""
使用 TorchTrainer 来启动训练，指定训练函数 train_func，设置训练的 epoch 数量为 5，使用 2 个工作节点（num_workers=2）。
配置失败处理（max_failures=1），即最多允许一次失败。
"""
local_dir = "/Users/xnpeng/sumoptis/sumo-rl/ray_results"

run_config = train.RunConfig(
    checkpoint_config=train.CheckpointConfig(
        num_to_keep=2,
    ),
    storage_path=local_dir,
    failure_config=train.FailureConfig(max_failures=1),
)

trainer = TorchTrainer(
    train_func,
    train_loop_config={"num_epochs": 5, "checkpoint_freq": 2},
    scaling_config=ScalingConfig(num_workers=2),
    run_config=run_config,
)
result = trainer.fit()

print("result-1:\n", result)
print("result.checkpoint=", result.checkpoint)
print("result.checkpoint.path=", result.checkpoint.path)


checkpoint = Checkpoint.from_directory(result.checkpoint.path)
trainer = TorchTrainer(
    train_func,
    train_loop_config={"num_epochs": 5},
    scaling_config=ScalingConfig(num_workers=2),
    resume_from_checkpoint=checkpoint,
)
result3 = trainer.fit()

print("result-3:\n", result3)
print("result.checkpoint=", result3.checkpoint)
print("result.checkpoint.path=", result3.checkpoint.path)

"""
恢复训练，结果不正确。说明“resume_from_checkpoint=checkpoint”这句不成功。

result-3:
 Result(
  metrics={},
  path='/Users/xnpeng/ray_results/TorchTrainer_2024-11-12_18-25-15/TorchTrainer_681c8_00000_0_2024-11-12_18-25-15',
  filesystem='local',
  checkpoint=None
)
result.checkpoint= None
Traceback (most recent call last):
  File "/Users/xnpeng/sumoptis/sumo-rl/ray_train/save_load_checkpoint_restore_3.py", line 146, in <module>
    print("result.checkpoint.path=", result3.checkpoint.path)
AttributeError: 'NoneType' object has no attribute 'path'

"""