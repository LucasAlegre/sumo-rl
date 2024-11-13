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


def load_model_from_checkpoint(checkpoint_path):
    # 创建新的模型实例
    model = nn.Linear(4, 1)
    model = ray.train.torch.prepare_model(model)
    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    # 加载模型权重
    model_state_dict = torch.load(os.path.join(checkpoint_path, "model.pt"), weights_only=True)
    model.module.load_state_dict(model_state_dict)
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), weights_only=True))
    start_epoch = (torch.load(os.path.join(checkpoint_path, "extra_state.pt"), weights_only=True)["epoch"] + 1)

    return model, optimizer, start_epoch


local_dir = "/Users/xnpeng/sumoptis/sumo-rl/ray_results"

run_config = train.RunConfig(
    checkpoint_config=train.CheckpointConfig(
        num_to_keep=2,
    ),
    storage_path=local_dir,
    failure_config=train.FailureConfig(max_failures=1),
)

train_loop_config = {"num_epochs": 50, "checkpoint_freq": 2}

trainer = TorchTrainer(
    train_func,
    train_loop_config=train_loop_config,
    scaling_config=ScalingConfig(num_workers=2),
    run_config=run_config,
)
result = trainer.fit()

print("result:\n", result)
print("result.checkpoint=", result.checkpoint)
print("result.metrics=", result.metrics)

df = result.metrics_dataframe
print("Minimum loss=", min(df["loss"]))

# Print available checkpoints
for checkpoint, metrics in result.best_checkpoints:
    print("Loss=", metrics["loss"], "checkpoint=", checkpoint)

# Get checkpoint with minimal loss
best_checkpoint = min(
    result.best_checkpoints, key=lambda checkpoint: checkpoint[1]["loss"]
)[0]

min_loss = min(
    result.best_checkpoints, key=lambda checkpoint: checkpoint[1]["loss"]
)[1]

print("MinimumLoss=:", min_loss, " BestCheckpoint=:", best_checkpoint)

"""


"""
