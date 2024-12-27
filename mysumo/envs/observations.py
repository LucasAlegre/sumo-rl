"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        logger.debug("@@@@@@@@@@DefaultObservationFunction.__call__ begin@@@@@@@@@@")
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        logger.debug("@@@@@@@@@@DefaultObservationFunction.__call__ end@@@@@@@@@@")
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

"""
这段代码定义了交通信号的观察函数。详细分析这个程序：

1. `ObservationFunction` 类：
   - 这是一个抽象基类，为所有观察函数定义了基本结构。
   - 包含两个抽象方法：`__call__()` 和 `observation_space()`，子类必须实现这两个方法。
   - 初始化时接收一个 `TrafficSignal` 对象，存储为 `self.ts`。

2. `DefaultObservationFunction` 类：
   - 继承自 `ObservationFunction`，实现了默认的观察函数。
   - `__call__()` 方法：
     a. 生成当前绿灯相位的独热编码（one-hot encoding）。
     b. 检查是否满足最小绿灯时间。
     c. 获取车道密度和队列长度。
     d. 将所有这些信息组合成一个 numpy 数组作为观察结果。
   - `observation_space()` 方法：
     定义了观察空间，使用 `gymnasium.spaces.Box`，范围在 0 到 1 之间。

3. 观察的组成：
   - 绿灯相位的独热编码：表示当前哪个绿灯相位是激活的。
   - 最小绿灯时间标志：表示是否已经满足最小绿灯时间要求。
   - 车道密度：表示每个车道上的车辆密度。
   - 车道队列：表示每个车道上等待的车辆数量。

4. 观察空间：
   - 维度为：绿灯相位数 + 1（最小绿灯时间标志）+ 2 * 车道数（密度和队列）
   - 所有值都被归一化到 0-1 范围内。

5. 特点：
   - 使用了 numpy 数组来高效地处理数值数据。
   - 观察结果被标准化，有利于神经网络的处理。
   - 包含了交通信号灯控制所需的关键信息：当前相位、时间约束、交通状况。

这个观察函数设计得很全面，包含了控制交通信号灯所需的主要信息。它提供了当前交通状况的快照，包括信号灯状态和道路使用情况，这对于训练强化学习代理来优化交通流非常有用。

使用这种观察函数，强化学习代理可以基于当前交通状况做出决策，例如何时切换信号灯相位，以最大化交通流量或最小化等待时间。


=======================================================

假设我们有一个十字路口，由一个交通信号灯控制。
这个信号灯有4个绿灯相位（南北直行，南北左转，东西直行，东西左转），4个进入路口的车道。

1. 观察空间：

首先，让我们计算观察空间的维度：
- 绿灯相位数：4
- 最小绿灯时间标志：1
- 车道数：4
- 总维度：4 (相位) + 1 (最小绿灯时间) + 2 * 4 (密度和队列) = 13

所以，观察空间将是一个13维的 Box 空间，每个维度的值在0到1之间：

```python
observation_space = spaces.Box(
    low=np.zeros(13, dtype=np.float32),
    high=np.ones(13, dtype=np.float32)
)
```

2. 观察示例：

现在，让我们假设在某个时刻，交通信号灯的状态如下：
- 当前是第2个绿灯相位（南北左转）
- 已经满足最小绿灯时间
- 车道密度：[0.3, 0.5, 0.2, 0.4]（分别对应南、北、东、西）
- 车道队列：[2, 3, 1, 2]（假设最大队列长度为10）

那么，观察将会是这样的：

```python
observation = np.array([
    0, 1, 0, 0,  # 绿灯相位的独热编码（第2个相位为1，其他为0）
    1,           # 最小绿灯时间标志（1表示已满足）
    0.3, 0.5, 0.2, 0.4,  # 车道密度
    0.2, 0.3, 0.1, 0.2   # 车道队列（归一化到0-1范围）
], dtype=np.float32)
```

解释：
- 前4个值 [0, 1, 0, 0] 表示当前是第2个绿灯相位。
- 第5个值 1 表示已经满足最小绿灯时间。
- 接下来的4个值 [0.3, 0.5, 0.2, 0.4] 表示4个车道的密度。
- 最后4个值 [0.2, 0.3, 0.1, 0.2] 表示4个车道的队列长度，这里假设最大队列长度为10，所以原始值 [2, 3, 1, 2] 被除以10归一化。

3. 使用示例：

在强化学习环境中，这个观察可能会这样使用：

```python
class TrafficEnvironment(gym.Env):
    def __init__(self):
        self.traffic_signal = TrafficSignal(...)
        self.observation_function = DefaultObservationFunction(self.traffic_signal)
        self.observation_space = self.observation_function.observation_space()
        self.action_space = spaces.Discrete(4)  # 4个可能的相位

    def step(self, action):
        # 执行动作（改变交通信号灯相位）
        self.traffic_signal.set_phase(action)
        
        # 模拟交通流动
        self.simulate_traffic()
        
        # 获取新的观察
        observation = self.observation_function()
        
        # 计算奖励
        reward = self.calculate_reward()
        
        # 检查是否结束
        done = self.is_done()
        
        return observation, reward, done, {}

    def reset(self):
        # 重置环境
        self.traffic_signal.reset()
        return self.observation_function()
```

在这个例子中，强化学习代理会接收到13维的观察向量，基于这个向量来决定下一步应该将交通信号切换到哪个相位，以优化交通流量。
这个观察提供了当前交通状况的全面信息，包括当前信号灯状态、是否可以切换相位、各个方向的交通密度和等待车辆数量，这些信息对于做出明智的交通控制决策非常重要。
"""