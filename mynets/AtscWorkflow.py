from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from mynets.envs.sumo_env import SumoEnv

# 创建环境
env_kwargs = {
    'net_file': "my-intersection/my-intersection.net.xml",
    "route_file": "my-intersection/my-intersection.rou.xml",
    "out_csv_name": "out/my-intersection.dqn",
    "single_agent": True,
    "use_gui": False,
    "num_seconds": 10000
}

env = make_vec_env(env_id='SumoEnv-v0', n_envs=5, env_kwargs=env_kwargs)


# env = RecordEpisodeStatistics(env)
# env = RecordVideo(env, video_folder="recording")

print("test env =======")
# 认识游戏环境
def test_env():
    print('env.observation_space=', env.observation_space)
    print('env.action_space=', env.action_space)

    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

    print('state=', state)
    print('action=', action)
    print('next_state=', next_state)
    print('reward=', reward)
    print('done=', done)
    print('info=', info)


test_env()

# 初始化模型
model = DQN(
    policy='MlpPolicy',
    env=env,
    # env=make_vec_env(env, n_envs=8),  # 使用N个环境同时训练
    learning_rate=1e-3,
    buffer_size=50000,  # 最多积累N步最新的数据,旧的删除
    learning_starts=2000,  # 积累了N步的数据以后再开始训练
    batch_size=64,  # 每次采样N步
    tau=0.8,  # 软更新的比例,1就是硬更新
    gamma=0.9,
    train_freq=(1, 'step'),  # 训练的频率
    target_update_interval=1000,  # target网络更新的频率
    policy_kwargs={},  # 网络参数
    verbose=0)

print("评估模型======")
# 评估模型
mean, std = evaluate_policy(model, env, n_eval_episodes=5, deterministic=False)
print(mean, std)

print("训练模型======")
# #训练模型
model.learn(10000, progress_bar=True)

print("保存模型======")
# #保存模型
model.save('model/my-intersection.dqn')
del model
env.close()

print("加载模型======")
# 加载模型
model = DQN.load('model/my-intersection.dqn')
env = model.get_env()
mean, std = evaluate_policy(model, env, n_eval_episodes=5, deterministic=False)
print(mean, std)


def test():
    state = env.reset()
    reward_sum = []
    over = False
    while not over:
        action, _ = model.predict(state)
        action = action.item()
        state, reward, over, _ = env.step(action)
        reward_sum.append(reward)

    print(sum(reward_sum), len(reward_sum), reward_sum)


print("测试模型======")
test()

del model
env.close()
