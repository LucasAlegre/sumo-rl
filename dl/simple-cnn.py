"""
PyTorch 是一个流行的开源机器学习库，广泛用于研究和开发。
在开始之前，我需要了解一些信息：
您希望解决什么类型的问题？（例如：图像分类、文本生成、回归分析等）
您是否有特定的数据集，还是希望我使用一个通用的公共数据集？
您是否有任何特定的网络架构或模型偏好？（例如：CNN、RNN、Transformer 等）
请提供这些细节，以便我为您创建一个定制的程序。

1，回归问题；2，公共数据集；3，CNN。

针对您的需求，我将编写一个使用卷积神经网络（CNN）解决回归问题的深度学习程序。
由于 CNN 主要用于处理图像数据，我将选择一个适合回归分析的图像数据集。
一个常用的选择是 Boston Housing Dataset，但由于它是基于非图像数据，我将使用另一个数据集，比如 MNIST 数据集，
并通过某种方式将其转换为回归问题（例如，预测图片中数字的大小）。

让我们开始编写这个程序。
首先，我将导入必要的库，设置数据集，定义 CNN 模型，然后进行训练和评估。

这个程序包括了以下几个部分：

SimpleCNN 类：定义了一个简单的卷积神经网络结构，用于处理图像并输出一个回归值。
数据加载和预处理：使用 torchvision 加载 MNIST 数据集，并应用了标准的预处理步骤。
转换为回归问题：通过修改标签将分类问题转换为回归问题。在这个例子中，我们简单地将标签（0-9）转换为浮点数（0.0-1.0）。
训练和评估函数：定义了训练和评估模型的函数。

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义一个简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # 输入通道为1，输出通道为16
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1) # 输出一个值作为回归结果

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleCNN()

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载并准备 MNIST 数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

# 测试集
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False)

# 转换数据集为回归问题
def convert_to_regression(target):
    # 在这里，我们简单地将目标转换为它的大小
    return target.float() / 9

# 训练模型的函数（由于环境限制，这里只提供代码）
def train(model, trainloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = convert_to_regression(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    print('Finished Training')

# 评估模型的性能（由于环境限制，这里只提供代码）
def evaluate(model, testloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            labels = convert_to_regression(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()

    print(f'Average loss on test set: {total_loss / len(testloader)}')

# 注意：以下两个函数在这个环境(GPT)中无法运行
train(model, trainloader, criterion, optimizer)
evaluate(model, testloader)

# 完整代码在您的本地环境中应该可以运行。您可以复制上述代码，然后在您的机器上运行它。
