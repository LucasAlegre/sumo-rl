import ray
from ray import train
import torch
import torch.nn as nn
from ray.air import ScalingConfig
from ray.train.torch import TorchTrainer


# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)


# 定义训练函数
def train_func(config):
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 这里放您的训练循环
    for epoch in range(config["num_epochs"]):
        # 您的训练代码
        train.report({"loss": 0.5})  # 报告指标


# 初始化Ray（如果还没有初始化）
ray.init()

# 创建训练器
trainer = TorchTrainer(
    train_func,
    train_loop_config={"num_epochs": 10},
    scaling_config=ScalingConfig(
        num_workers=4,
        use_gpu=False,
        trainer_resources={"CPU": 1},
        placement_strategy="SPREAD",
    )
)

# 开始训练
results = trainer.fit()
print(f"Training completed. Final results: {results.metrics}")
