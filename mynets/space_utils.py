from gymnasium.spaces import Box, Discrete
import numpy as np


# 定义一个函数，将离散动作映射到连续动作空间
def map_discrete_to_continuous(discrete_space, discrete_action, min_value, max_value):
    num_actions = discrete_space.n
    return np.array([min_value + (max_value - min_value) * discrete_action / (num_actions - 1)])


# 定义一个函数，将连续动作映射到离散动作空间
def map_continuous_to_discrete(continuous_space, continuous_action, num_actions):
    continuous_high = continuous_space.high
    continuous_low = continuous_space.low
    return int(continuous_action * (num_actions - 1) / (continuous_high - continuous_low))


def test_1():
    # 创建一个离散动作空间
    discrete_action_space = Discrete(4)

    # 离散动作空间大小
    num_actions = discrete_action_space.n

    # 设定连续动作空间的范围
    min_value = 0.0
    max_value = 1.0

    # 创建连续动作空间
    continuous_action_space = Box(low=min_value, high=max_value, shape=(1,))

    # 测试映射函数
    for action in range(num_actions):
        continuous_action = map_discrete_to_continuous(discrete_action_space, action, min_value, max_value)
        print(f"Discrete action {action}: maps to continuous action {continuous_action}")

    # 示例：如何使用映射函数
    # 假设你有一个离散动作
    discrete_action = 2
    # 使用映射函数将其转换为连续动作
    continuous_action = map_discrete_to_continuous(discrete_action_space, discrete_action, min_value, max_value)
    print(f"Discrete action {discrete_action} maps to continuous action {continuous_action}")


def test_2():
    # 创建一个连续动作空间
    continuous_action_space = Box(low=0.0, high=1.0, shape=(1,))

    # 定义离散动作空间的大小
    num_actions = 5

    # 创建离散动作空间
    discrete_action_space = Discrete(num_actions)

    # 测试映射函数
    for action in np.linspace(continuous_action_space.low, continuous_action_space.high, 10):
        discrete_action = map_continuous_to_discrete(continuous_action_space, action, num_actions)
        print(f"Continuous action {action}: maps to discrete action {discrete_action}")

test_1()

test_2()