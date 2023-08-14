from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
# 导入模型
model = DQN.load("./model/CartPole.pkl")

state = env.reset()
done = False
score = 0
while not done:
    # 预测动作
    action, _ = model.predict(observation=state)
    # 与环境互动
    state, reward, done, truncated, info = env.step(action=action)
    score += reward
    env.render()
env.close()
print("score=", score)
