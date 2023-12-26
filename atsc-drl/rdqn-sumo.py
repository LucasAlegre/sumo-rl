import os
import sys
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

# 设置SUMO仿真环境
sumoBinary = "sumo"  # SUMO可执行文件路径
sumoCmd = [sumoBinary, "-c", "net/single-intersection.sumocfg"]  # SUMO配置文件路径

# 定义状态和动作空间
state_size = 4  # 状态空间维度
action_size = 2  # 动作空间维度


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # 经验回放池
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索因子
        self.epsilon_decay = 0.995  # 探索因子的衰减率
        self.epsilon_min = 0.01  # 探索因子的最小值
        self.learning_rate = 0.001  # 学习率
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()  # 神经网络模型
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # 输入层，24个神经元，ReLU激活函数，状态空间维度个输入
        model.add(Dense(24, activation='relu'))  # 隐藏层   24个神经元，ReLU激活函数
        model.add(Dense(self.action_size, activation='linear'))  # 输出层，动作空间维度个神经元，线性激活函数
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  # 编译模型，损失函数：均方误差，优化器：Adam
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # 将经验样本存储到经验回放池中

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # 探索因子大于随机数
            return np.random.choice(self.action_size)  # 探索
        act_values = self.model.predict(state)  # 利用模型预测Q值
        return np.argmax(act_values[0])  # 返回Q值最大的动作

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)  # 从经验回放池中随机选择一批经验样本
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]  # 获取经验样本
            target = reward  # 计算目标Q值
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))  # 利用贝尔曼方程计算目标Q值
            target_f = self.model.predict(state)  # 利用模型预测Q值
            target_f[0][action] = target  # 更新目标Q值
            self.model.fit(state, target_f, epochs=1, verbose=0)  # 训练模型
        if self.epsilon > self.epsilon_min:  # 如果探索因子大于最小值
            self.epsilon *= self.epsilon_decay  # 更新探索因子


def calculate_reward():
    """
    vi(t): 车辆i在仿真时间步t的速度
    wi(t): 车辆i在仿真时间步t的等待时间；= 0, if vi(t) <= 2 m/s; += 1, if vi(t) > 2 m/s
    ri = c - c * (wi/wm)^2
    c: 常数，控制ri上限
    wi: 车辆i的等待时间
    wm: 车辆等待时间的阀值
    """
    return traci.edge.getLastStepVehicleNumber("inflow") - traci.edge.getLastStepVehicleNumber("outflow")  # 计算奖励


# 初始化交通信号控制代理
agent = DQNAgent(state_size, action_size)

# 连接到SUMO仿真环境
traci.start(sumoCmd)

# 训练代理
batch_size = 32
num_episodes = 1000
for episode in range(num_episodes):  # 遍历训练次数
    state = traci.trafficlight.getRedYellowGreenState("traffic_light_id")  # 获取当前状态
    state = np.reshape(state, [1, state_size])  # 将状态转换为神经网络的输入形式
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)  # 根据当前状态选择动作
        traci.trafficlight.setRedYellowGreenState("traffic_light_id", action)  # 执行动作
        traci.simulationStep()  # 执行一个仿真步长
        next_state = traci.trafficlight.getRedYellowGreenState("traffic_light_id")  # 获取下一个状态
        next_state = np.reshape(next_state, [1, state_size])  # 将状态转换为神经网络的输入形式
        reward = calculate_reward()  # 根据具体的奖励函数计算奖励
        agent.remember(state, action, reward, next_state, done) # 记录经验
        state = next_state  # 更新状态
        total_reward += reward  # 更新总奖励
    if len(agent.memory) > batch_size:  # 如果经验回放缓冲区中的经验数量大于批大小
        agent.replay(batch_size)    # 从经验回放缓冲区中随机选择一批经验样本进行训练
    print("Episode: {}, Total Reward: {}".format(episode, total_reward))    # 打印训练次数和总奖励

# 关闭SUMO仿真环境
traci.close()

# 使用训练好的代理进行测试
test_episodes = 10
for episode in range(test_episodes):    # 遍历测试次数
    state = traci.trafficlight.getRedYellowGreenState("traffic_light_id")  # 获取当前状态
    state = np.reshape(state, [1, state_size])  # 将状态转换为神经网络的输入形式
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        traci.trafficlight.setRedYellowGreenState("traffic_light_id", action)
        traci.simulationStep()
        next_state = traci.trafficlight.getRedYellowGreenState("traffic_light_id")
        next_state = np.reshape(next_state, [1, state_size])
        reward = calculate_reward()  # 根据具体的奖励函数计算奖励
        state = next_state
        total_reward += reward
    print("Test Episode: {}, Total Reward: {}".format(episode, total_reward))
