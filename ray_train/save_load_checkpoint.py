import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from ray.air import RunConfig, CheckpointConfig
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
    # Wrap the model in DDP
    model = ray.train.torch.prepare_model(nn.Linear(4, 1))
    criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=3e-4)
    for epoch in range(config["num_epochs"]):
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
                torch.save(
                    model.module.state_dict(),  # NOTE: Unwrap the model.
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report(metrics, checkpoint=checkpoint)

local_dir = "/Users/xnpeng/sumoptis/sumo-rl/ray_results"
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
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
