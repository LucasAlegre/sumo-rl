import os
import tempfile

import torch
import torch.nn as nn
from ray import train, tune
from ray.train import Checkpoint
from torch import optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_func(config):
    start = 1
    my_model = SimpleNN()
    # 定义损失函数（MSE损失函数）和优化器（使用Adam优化器）
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(my_model.parameters(), lr=0.001)  # Adam优化器

    # 假设有一组输入数据和目标标签
    input_data = torch.randn(32, 10)  # 32个样本，每个样本有10个特征
    target_data = torch.randn(32, 1)  # 32个样本，每个样本有一个目标值

    checkpoint = train.get_checkpoint()  #  从框架中获取checkpoint
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            # 从checkpoint中加载环境字典
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start = checkpoint_dict["epoch"] + 1
            # 将checkpoint中的模型状态model_state字典装载到模型中my_model
            my_model.load_state_dict(checkpoint_dict["model_state"])

    for epoch in range(start, config["epochs"] + 1):
        my_model.train()  # 设置模型为训练模式
        output = my_model(input_data)  # 前向传播
        loss = criterion(output, target_data)  # 计算损失
        optimizer.zero_grad()  # 清除旧的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        metrics = {"loss": loss.item()}
        with tempfile.TemporaryDirectory() as tempdir:
            # 保存检查点
            torch.save(
                {"epoch": epoch, "model_state": my_model.state_dict()},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            # 报告metrics & checkpoint给主训练程序(调度程序)
            train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))


tuner = tune.Tuner(train_func, param_space={"epochs": 5})
result_grid = tuner.fit()

print(result_grid)