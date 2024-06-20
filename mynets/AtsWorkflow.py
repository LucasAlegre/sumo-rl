import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from sumo_rl import SumoEnvironment


class SumoEnvWrapper(gym.Wrapper):
    def __init__(self,
                 net_file,
                 route_file,
                 out_cvs_name,
                 single_agent=True,
                 use_gui=False,
                 num_seconds=10000,
                 render_mode='rgb_array',
                 video_path="video",
                 monitor_path="monitor"):
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=out_cvs_name,
            single_agent=single_agent,
            use_gui=use_gui,
            num_seconds=num_seconds,
            render_mode=render_mode
        )
        env = RecordEpisodeStatistics(env)
        env = RecordVideo(env, video_folder=video_path)
        env = Monitor(env, monitor_path)
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self, *, seed=None, options=None):
        state, info = self.env.reset()
        self.step_n = 0
        return state, info

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)
        self.step_n += 1
        if self.step_n >= 200:
            done = True
        return state, reward, done, trunc, info


net_file = "my-intersection/my-intersection.net.xml",
route_file = "my-intersection/my-intersection.rou.xml",
out_csv_name = "out/my-intersection-dqn",
env = SumoEnvWrapper(net_file, route_file, out_csv_name)
env.reset()


# 认识游戏环境
def test_env():
    print('env.observation_space=', env.observation_space)
    print('env.action_space=', env.action_space)

    state, info = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, trunc, info = env.step(action)

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
    env=make_vec_env(SumoEnvWrapper, n_envs=8),  # 使用N个环境同时训练
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

mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=False)
print(mean, std)

# #训练模型
model.learn(50_0000, progress_bar=True)

# #保存模型
model.save('model/my-intersection.dqn')

# 加载模型
model = DQN.load('model/my.intersection.dqn')

mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=False)
print(mean, std)


def test():
    state, info = env.reset()
    reward_sum = []
    over = False
    while not over:
        action, _ = model.predict(state)
        action = action.item()
        state, reward, over, _, _ = env.step(action)
        reward_sum.append(reward)

    print(sum(reward_sum), len(reward_sum), reward_sum)


test()
