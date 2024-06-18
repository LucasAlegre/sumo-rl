# 字符到one-hot编码的映射
import numpy as np
import torch
from torch import nn

char_to_onehot = {
    'G': [1, 0, 0, 0],
    'Y': [0, 1, 0, 0],
    'r': [0, 0, 1, 0],
    'R': [0, 0, 0, 1]
}


def state_to_onehot(state):
    # 将状态字符串转换为one-hot编码矩阵
    onehot = [char_to_onehot[char] for char in state]
    return np.array(onehot).flatten()  # 展平矩阵


class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)  # 输入层，24个神经元
        self.fc2 = nn.Linear(24, 24)  # 隐藏层，24个神经元
        self.fc3 = nn.Linear(24, action_size)  # 输出层，动作空间维度个神经元

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = torch.relu(self.fc2(x))  # ReLU 激活函数
        x = self.fc3(x)  # 线性激活函数
        return x


state_size = 16
action_size = 2

state = "GGrr"
state = state_to_onehot(state)
state = torch.FloatTensor(state).view(1, -1)
model = NeuralNetwork(state_size, action_size)
print(model)

for i in range(10):
    act_value = model(state)
    print(act_value)
    action_numpy = act_value.detach().numpy()
    print(action_numpy)
    action = np.argmax(act_value.detach().numpy()[0])
    print(action)
