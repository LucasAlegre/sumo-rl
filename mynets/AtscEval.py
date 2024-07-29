import sys
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.dqn import DQN
sys.path.append('..')

from mysumo.envs.sumo_env import SumoEnv, ContinuousSumoEnv

# 没有写完。想将整个流程分为几部分：（1）训练前评估性能；（2）训练模型，num_seconds：每回合时间（秒），n_eval_episodes：评估回合，total_timesteps：总训练步，n_steps：每回合时间步。
env = SumoEnv(
    net_file="my-intersection/my-intersection.net.xml",
    route_file="my-intersection/my-intersection.rou.xml",
    single_agent=True,
    use_gui=False,
    num_seconds=20000,  # 仿真秒，最大值20000
    render_mode="rgb_array",  # 'rgb_array':This system has no OpenGL support.
)

env = Monitor(env, "monitor/SumoEnv-DQN")
env = DummyVecEnv([lambda: env])

agent = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=0.001,
    learning_starts=0,
    train_freq=1,
    target_update_interval=1000,
    exploration_initial_eps=0.05,
    exploration_final_eps=0.01,
    tensorboard_log="log/my-intersection-dqn-log",
    verbose=1,
)

# 加载训练好的模型
agent = DQN.load("model/my-intersection-dqn.zip")

env = agent.get_env()
obs = env.reset()
# 运行仿真
for i in range(10):
    # 获取当前状态,使用模型选择动作
    action, state = agent.predict(obs)  # 实现这个函数来获取当前交叉路口状态

    # 在环境中执行动作,并收集性能指标
    obs, reward, dones, info = env.step(action)

    # 收集性能指标
    print("\n======", i, "======")
    print(" info:", info)

# 结束仿真
env.close()
del agent

# 可以进一步绘制图表、进行统计分析等
