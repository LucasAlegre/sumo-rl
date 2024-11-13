import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import ray.train.torch
from ray import train
from ray.train import Checkpoint, ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer


def train_func(config):
    n = 100
    # create a toy dataset
    # data   : X - dim = (n, 4)
    # target : Y - dim = (n, 1)
    X = torch.Tensor(np.random.normal(0, 1, size=(n, 4)))
    Y = torch.Tensor(np.random.uniform(0, 1, size=(n, 1)))
    # toy neural network : 1-layer
    # Wrap the model in DDP - 分布式数据并发模型
    model = ray.train.torch.prepare_model(nn.Linear(4, 1))
    criterion = nn.MSELoss()  # 损失函数
    optimizer = Adam(model.parameters(), lr=3e-4)  # 参数优化器

    for epoch in range(config["num_epochs"]):
        y = model.forward(X)  # 前向传播
        loss = criterion(y, Y)  # 损失值
        optimizer.zero_grad()
        loss.backward()  # 后向传播
        optimizer.step()  # 优化一步

        metrics = {"loss": loss.item()}  # 训练指标

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            should_checkpoint = epoch % config.get("checkpoint_freq", 1) == 0
            # In standard DDP training, where the model is the same across all ranks,
            # only the global rank 0 worker needs to save and report the checkpoint
            if train.get_context().get_world_rank() == 0 and should_checkpoint:
                torch.save(
                    model.module.state_dict(),  # NOTE: Unwrap the model.
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report(metrics, checkpoint=checkpoint)


local_dir = "/Users/xnpeng/sumoptis/sumo-rl/ray_results"

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="loss",
        checkpoint_score_order="min",
    ),
    storage_path=local_dir,
)

trainer = TorchTrainer(
    train_func,
    train_loop_config={"num_epochs": 5},
    scaling_config=ScalingConfig(num_workers=2),
    run_config=run_config,
)
result = trainer.fit()

print("result:\n", result)

print("result.metrics=", result.metrics)
print("result.metrics['loss']=", result.metrics["loss"])
print("result.path=", result.path)
print("result.filesystem=", result.filesystem)
print("result.checkpoint=", result.checkpoint)
print("result.checkpoint.path=", result.checkpoint.path)

# 训练后使用checkpoint:
example_checkpoint_dir = result.checkpoint.path
# Create the checkpoint, which is a reference to the directory.
checkpoint = Checkpoint.from_directory(example_checkpoint_dir)  # 从路径名(不含model.pt)加载checkpoint

print("\ncheckpoint:\n", checkpoint)

# Inspect the checkpoint's contents with either `as_directory` or `to_directory`:
with checkpoint.as_directory() as checkpoint_dir:  # 作为文件夹，其中存在文件model.pt
    assert Path(checkpoint_dir).joinpath("model.pt").exists()
    checkpoint_path = Path(checkpoint_dir).joinpath("model.pt")
    print("checkpoint_path:\n", checkpoint_path)

checkpoint_dir = checkpoint.to_directory()  # 转变为文件夹，其中存在文件model.pt
assert Path(checkpoint_dir).joinpath("model.pt").exists()
print("checkpoint_dir:\n", checkpoint_dir)

"""
运行结果：
result:
 Result(
  metrics={'loss': 1.6687583923339844},
  path='/Users/xnpeng/sumoptis/sumo-rl/ray_results/TorchTrainer_2024-11-12_16-44-58/TorchTrainer_66977_00000_0_2024-11-12_16-44-59',
  filesystem='local',
  checkpoint=Checkpoint(filesystem=local, path=/Users/xnpeng/sumoptis/sumo-rl/ray_results/TorchTrainer_2024-11-12_16-44-58/TorchTrainer_66977_00000_0_2024-11-12_16-44-59/checkpoint_000004)
)
result.metrics= {'loss': 1.6687583923339844, 'timestamp': 1731401103, 'checkpoint_dir_name': 'checkpoint_000004', 'should_checkpoint': True, 'done': True, 'training_iteration': 5, 'trial_id': '66977_00000', 'date': '2024-11-12_16-45-03', 'time_this_iter_s': 0.0011529922485351562, 'time_total_s': 2.000298023223877, 'pid': 15829, 'hostname': 'apen.local', 'node_ip': '127.0.0.1', 'config': {'train_loop_config': {'num_epochs': 5}}, 'time_since_restore': 2.000298023223877, 'iterations_since_restore': 5, 'experiment_tag': '0'}
result.metrics['loss']= 1.6687583923339844
result.path= /Users/xnpeng/sumoptis/sumo-rl/ray_results/TorchTrainer_2024-11-12_16-44-58/TorchTrainer_66977_00000_0_2024-11-12_16-44-59
result.filesystem= <pyarrow._fs.LocalFileSystem object at 0x11ea841f0>
result.checkpoint= Checkpoint(filesystem=local, path=/Users/xnpeng/sumoptis/sumo-rl/ray_results/TorchTrainer_2024-11-12_16-44-58/TorchTrainer_66977_00000_0_2024-11-12_16-44-59/checkpoint_000004)
result.checkpoint.path= /Users/xnpeng/sumoptis/sumo-rl/ray_results/TorchTrainer_2024-11-12_16-44-58/TorchTrainer_66977_00000_0_2024-11-12_16-44-59/checkpoint_000004

checkpoint:
 Checkpoint(filesystem=local, path=/Users/xnpeng/sumoptis/sumo-rl/ray_results/TorchTrainer_2024-11-13_07-23-12/TorchTrainer_179b8_00000_0_2024-11-13_07-23-15/checkpoint_000004)
checkpoint_path:
 /Users/xnpeng/sumoptis/sumo-rl/ray_results/TorchTrainer_2024-11-13_07-23-12/TorchTrainer_179b8_00000_0_2024-11-13_07-23-15/checkpoint_000004/model.pt
checkpoint_dir:
 /var/folders/yk/xs1byz1x1hdbbwk36sy03jym0000gn/T/checkpoint_tmp_7312a8a7507e4e9b8eef191dcb732501

"""