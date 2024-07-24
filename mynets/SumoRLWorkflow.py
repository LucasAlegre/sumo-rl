import sys
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append('..')

import mysumo.envs  # 确保自定义环境被注册

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from mysumo.envs.sumo_env import ContinuousSumoEnv


# 创建SUMO-RL环境
def make_env():
    return ContinuousSumoEnv(net_file='my-intersection/my-intersection.net.xml',
                             route_file='my-intersection/my-intersection.rou.xml',
                             use_gui=False,
                             num_seconds=3600,
                             delta_time=5,
                             yellow_time=2,
                             min_green=5,
                             max_green=50,
                             reward_fn='diff-waiting-time'
                             )


# 将环境包装在DummyVecEnv中
env = DummyVecEnv([make_env])

# 创建SAC模型
model = SAC("MlpPolicy", env, verbose=1)

# 训练模型
total_timesteps = 100000
model.learn(total_timesteps=total_timesteps)

# 保存训练好的模型
model.save("sac_sumo_model")

# 评估模型
episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
    print(f"Episode {ep + 1}: Total Reward = {total_reward}")

# 关闭环境
env.close()
