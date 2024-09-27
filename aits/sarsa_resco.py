import os
import sys

import fire

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

from sumo_rl import grid4x4, arterial4x4, cologne1, cologne3, cologne8, ingolstadt1, ingolstadt7, ingolstadt21


def run(use_gui=True, episodes=50):
    fixed_tl = False

    # env = grid4x4(out_csv_name="outputs/grid4x4/grid4x4", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    env = arterial4x4(out_csv_name="outputs/grid4x4/arterial4x4", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = cologne1(out_csv_name="outputs/grid4x4/cologne1", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = cologne3(out_csv_name="outputs/grid4x4/cologne3", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = cologne8(out_csv_name="outputs/grid4x4/cologne8", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = ingolstadt1(out_csv_name="outputs/grid4x4/ingolstadt1", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = ingolstadt7(out_csv_name="outputs/grid4x4/ingolstadt7", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)
    # env = ingolstadt21(out_csv_name="outputs/grid4x4/ingolstadt21", use_gui=use_gui, yellow_time=2, fixed_ts=fixed_tl)

    env.reset()

    agents = {
        ts_id: TrueOnlineSarsaLambda(
            env.observation_spaces[ts_id],
            env.action_spaces[ts_id],
            alpha=0.0001,
            gamma=0.95,
            epsilon=0.05,
            lamb=0.1,
            fourier_order=7,
        )
        for ts_id in env.agents
    }

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = {agent: False for agent in env.agents}

        if fixed_tl:
            while env.agents:
                _, _, terminated, truncated, _ = env.step(None)
        else:
            while env.agents:
                actions = {ts_id: agents[ts_id].act(obs[ts_id]) for ts_id in obs.keys()}
                next_obs, r, terminated, truncated, _ = env.step(actions=actions)

                for ts_id in next_obs.keys():
                    agents[ts_id].learn(
                        state=obs[ts_id],
                        action=actions[ts_id],
                        reward=r[ts_id],
                        next_state=next_obs[ts_id],
                        done=terminated[ts_id],
                    )
                    obs[ts_id] = next_obs[ts_id]

    env.close()


if __name__ == "__main__":
    fire.Fire(run)

# 测试说明 2023-10-30
# 1. 检查环境：libs-mac/ubuntu-2023-10-30.txt
# 2. python experiments/sarsa_resco.py
#    ubuntu运行成功，macOS运行正常。

"""
分析这个程序的执行逻辑:

1. 导入模块:
   - 程序首先导入必要的模块,包括SUMO工具和自定义的强化学习算法TrueOnlineSarsaLambda。

2. 主要函数run:
   - 创建交通环境(env),默认使用arterial4x4环境。
   - 为每个交通信号灯(ts_id)创建一个TrueOnlineSarsaLambda代理。
   - 进行多轮训练:
     - 每轮开始时重置环境。
     - 如果不是固定时间控制(fixed_tl为False):
       - 循环执行直到环境结束:
         - 每个代理根据当前观察选择动作
         - 执行动作,获取新状态、奖励等信息
         - 每个代理使用这些信息学习
         - 更新当前观察
     - 如果是固定时间控制,只让环境运行,不进行学习。

3. 主函数:
   - 使用fire库处理命令行参数,调用run函数。

4. 测试说明:
   - 提供了运行脚本的环境要求和命令。

这个程序实现了一个交通信号控制的强化学习方案,使用SARSA(λ)算法训练每个交通信号灯的控制策略,目标是优化交通流量。
程序可以在不同交通环境中运行,通过注释/取消注释相应行来选择环境。

==================================================

这个程序中多智能体学习的原理:

1. 多智能体设置:
   - 每个交通信号灯被视为一个独立的智能体。
   - 使用字典 `agents` 存储所有智能体,键为交通信号灯ID,值为对应的TrueOnlineSarsaLambda代理。

2. 学习算法:
   - 每个智能体使用True Online SARSA(λ)算法,这是一种在线策略时序差分学习方法。
   - 算法参数包括学习率(alpha)、折扣因子(gamma)、探索率(epsilon)和λ值。

3. 状态空间和动作空间:
   - 每个智能体的观察空间和动作空间由环境定义。
   - 使用傅里叶基函数进行特征提取,order设为7。

4. 交互与学习过程:
   - 在每个时间步:
     - 所有智能体同时根据各自的观察选择动作。
     - 环境接收所有动作,返回新的观察、奖励等信息。
     - 每个智能体独立学习,更新其价值函数和策略。

5. 分散式学习:
   - 每个智能体仅基于自己的观察和奖励进行学习,不直接共享信息。
   - 这种方法使系统具有良好的可扩展性,适用于大规模交通网络。

6. 间接协作:
   - 虽然智能体独立学习,但它们的行为会影响整体交通状况。
   - 通过环境的反馈,智能体可以间接地学习到协作行为。

7. 探索与利用:
   - epsilon-贪婪策略用于平衡探索和利用。
   - 智能体有小概率(epsilon)随机选择动作,以探索新策略。

这种多智能体强化学习方法允许每个交通信号灯适应性地学习最优控制策略,同时考虑到整体交通网络的复杂动态。

==================================================

Q: 请问，能否使用其他算法替代TrueOnlineSarsaLambda?

A: 是的,完全可以使用其他算法替代TrueOnlineSarsaLambda。以下是一些可能的替代选择:

1. Q-learning:
   - 这是一种经典的离线策略算法,易于实现且广泛使用。
   - 适合离散状态和动作空间。

2. DQN (Deep Q-Network):
   - 使用深度神经网络来近似Q函数,适合处理高维状态空间。
   - 可以处理连续状态空间,但仍限于离散动作空间。

3. DDPG (Deep Deterministic Policy Gradient):
   - 适用于连续动作空间的算法。
   - 结合了DQN和策略梯度方法的优点。

4. PPO (Proximal Policy Optimization):
   - 一种流行的策略梯度方法,在稳定性和性能方面表现良好。
   - 适用于连续和离散动作空间。

5. A3C (Asynchronous Advantage Actor-Critic):
   - 异步方法,可以有效利用多核处理器。
   - 适合大规模并行训练。

6. SAC (Soft Actor-Critic):
   - 结合了最大熵强化学习的思想,在探索-利用平衡方面表现出色。
   - 适用于连续动作空间。

替换算法时,需要注意:

1. 修改代理初始化部分,使用新算法的类。
2. 调整学习过程,以匹配新算法的API。
3. 可能需要重新设计状态表示和奖励函数。
4. 考虑新算法的特点,如是否适合多智能体设置、计算复杂度等。

选择替代算法时,应考虑问题的特性(如状态/动作空间的连续性)、计算资源、以及对算法性能和稳定性的要求。

"""