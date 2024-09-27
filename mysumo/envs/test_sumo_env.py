import os
import sys

sys.path.append('../..')

import mysumo
import mysumo.envs  # 确保自定义环境被注册
from mysumo import arterial4x4

# 创建环境
env = arterial4x4(out_csv_name="outputs/grid4x4/arterial4x4", use_gui=True, yellow_time=2, fixed_ts=False)

# 运行 10 个 episode
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 这里应该有一个策略来选择动作，这里我们随机选择
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # 可以打印一些信息
        print(f"Step info: {info}")
    
    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

# 关闭环境
env.close()

# 保存 CSV 数据
env.save_csv("final_results", episode + 1)