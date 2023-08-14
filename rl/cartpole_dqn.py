from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
# 把环境向量化，如果有多个环境写成列表传入DummyVecEnv中，可以用一个线程来执行多个环境，提高训练效率
env = DummyVecEnv([lambda: env])
# 定义一个DQN模型，设置其中的各个参数
model = DQN(
    "MlpPolicy",  # MlpPolicy定义策略网络为MLP网络
    env=env,
    learning_rate=5e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    target_update_interval=250,
    policy_kwargs={"net_arch": [256, 256]},  # 这里代表隐藏层为2层256个节点数的网络
    verbose=1,  # verbose=1代表打印训练信息，如果是0为不打印，2为打印调试信息
    tensorboard_log="./tensorboard/CartPole-v1/"  # 训练数据保存目录，可以用tensorboard查看
)
# 开始训练
model.learn(total_timesteps=100000)
# 策略评估，可以看到倒立摆在平稳运行了
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
# env.close()
print("mean_reward:", mean_reward, "std_reward:", std_reward)
# 保存模型到相应的目录
model.save("./model/CartPole.pkl")

# 导入模型
model2 = DQN.load("./model/CartPole.pkl")

state = env.reset()
done = False
score = 0
while not done:
    # 预测动作
    action, _ = model2.predict(observation=state)
    # 与环境互动
    state, reward, done, truncated, info = env.step(action=action)
    score += reward
    env.render()
env.close()
print("score=", score)
