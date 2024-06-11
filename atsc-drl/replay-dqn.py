import gymnasium as gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import Adam

"""
有经验回放的DQN算法

有经验回放的DQN算法（Deep Q-Network with Experience Replay）是一种深度强化学习算法，用于解决基于值函数的强化学习问题。它结合了深度神经网络和经验回放的概念，以提高学习效率和稳定性。
在传统的Q-learning算法中，智能体通过与环境交互来学习最优策略。然而，这种方法容易受到数据的相关性和不稳定性的影响。为了解决这个问题，DQN算法引入了经验回放的机制。
经验回放是一种存储和重用智能体与环境交互的经验的方法。智能体在与环境交互时，将每个状态转换、动作、奖励和下一个状态存储在经验回放缓冲区中。然后，智能体从缓冲区中随机选择一批经验样本进行训练，而不是立即使用当前的经验样本进行学习。这样做的好处是可以减少数据的相关性，提高训练的效率和稳定性。
DQN算法使用深度神经网络来近似值函数，将状态作为输入，输出每个动作的Q值。通过使用经验回放的样本进行训练，DQN算法可以通过最小化预测Q值与目标Q值之间的差异来更新神经网络的权重。目标Q值是通过使用贝尔曼方程计算得到的，它考虑了当前状态的即时奖励和下一个状态的最大Q值。
通过使用经验回放的DQN算法，智能体可以从之前的经验中学习，并且可以更好地处理数据的相关性和不稳定性。这使得算法更加高效和稳定，能够在复杂的强化学习任务中取得更好的性能。
"""


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 状态空间的维度
        self.action_size = action_size  # 动作空间的维度
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索因子
        self.epsilon_decay = 0.995  # 探索因子的衰减率
        self.epsilon_min = 0.01  # 探索因子的最小值
        self.learning_rate = 0.001  # 学习率
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()  # 创建一个序贯模型
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # 添加全连接层, 输入层: 24个神经元, 激活函数:relu
        model.add(Dense(24, activation='relu'))  # 添加全连接层, 隐藏层: 24个神经元, 激活函数:relu
        model.add(Dense(self.action_size, activation='linear'))  # 添加全连接层, 输出层: 动作空间的维度, 激活函数:linear
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  # 编译模型, 损失函数: mse, 优化器: Adam
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # 探索因子大于随机数
            return np.random.choice(self.action_size)  # 探索
        act_values = self.model.predict(state)  # 利用模型预测Q值
        return np.argmax(act_values[0])  # 返回Q值最大的动作

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)  # 从经验回放缓冲区中随机选择一批经验样本
        for idx in minibatch:  # 遍历经验样本
            state, action, reward, next_state, done = self.memory[idx]  # 获取经验样本
            target = reward  # 计算目标Q值
            if not done:  # 如果不是终止状态
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))  # 利用贝尔曼方程计算目标Q值
            target_f = self.model.predict(state)  # 利用模型预测Q值
            target_f[0][action] = target  # 更新目标Q值
            self.model.fit(state, target_f, epochs=1, verbose=0)  # 训练模型
        if self.epsilon > self.epsilon_min:  # 如果探索因子大于最小值
            self.epsilon *= self.epsilon_decay  # 更新探索因子


# 初始化环境和代理
env = gym.make('CartPole-v1')  # 初始化环境
state_size = env.observation_space.shape[0]  # 状态空间的维度
action_size = env.action_space.n  # 动作空间的维度
agent = DQNAgent(state_size, action_size)  # 初始化代理

# 训练代理
batch_size = 32  # 批大小
num_episodes = 1000  # 训练次数
for episode in range(num_episodes):  # 遍历训练次数
    state, info = env.reset()  # 初始化环境
    state = np.reshape(state, [1, state_size])  # 将状态转换为输入向量
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)  # 代理根据当前状态选择动作
        next_state, reward, done, trunc, _ = env.step(action)  # 与环境交互
        next_state = np.reshape(next_state, [1, state_size])  # 将下一个状态转换为输入向量
        agent.remember(state, action, reward, next_state, done)  # 记录经验
        state = next_state  # 更新状态
        total_reward += reward  # 更新总奖励
    if len(agent.memory) > batch_size:  # 如果经验回放缓冲区中的经验数量大于批大小
        agent.replay(batch_size)  # 从经验回放缓冲区中随机选择一批经验样本进行训练
    print("Episode: {}, Total Reward: {}".format(episode, total_reward))  # 打印训练次数和总奖励

"""
result:

Episode: 990, Total Reward: 2592.0
Episode: 991, Total Reward: 146.0
Episode: 992, Total Reward: 156.0
Episode: 993, Total Reward: 170.0
Episode: 994, Total Reward: 202.0
Episode: 995, Total Reward: 341.0
Episode: 996, Total Reward: 200.0
Episode: 997, Total Reward: 199.0
Episode: 998, Total Reward: 178.0
Episode: 999, Total Reward: 163.0
"""

# 使用训练好的代理进行测试
test_episodes = 10
for episode in range(test_episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        total_reward += reward
    print("Test Episode: {}, Total Reward: {}".format(episode, total_reward))


"""
result:
Test Episode: 0, Total Reward: 1336.0
Test Episode: 1, Total Reward: 204.0
Test Episode: 2, Total Reward: 268.0
Test Episode: 3, Total Reward: 185.0
Test Episode: 4, Total Reward: 283.0
Test Episode: 5, Total Reward: 625.0
Test Episode: 6, Total Reward: 190.0
Test Episode: 7, Total Reward: 434.0
Test Episode: 8, Total Reward: 162.0
Test Episode: 9, Total Reward: 357.0
"""