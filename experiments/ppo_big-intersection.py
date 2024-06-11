import os
import sys

import gymnasium as gym

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file="nets/big-intersection/big-intersection.net.xml",
    single_agent=True,
    route_file="nets/big-intersection/routes.rou.xml",
    out_csv_name="outputs/big-intersection/ppo",
    use_gui=True,
    num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
    render_mode="human"
)

model = PPO(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    verbose=1,
    tensorboard_log="./tensorboard/big-intersection/",
)
model.learn(total_timesteps=100000)

# Save, load, evaluate and predict the model
model.save("./model/ppo_big-intersection")
del model

model = PPO.load("./model/ppo_big-intersection", env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print("mean_reward:", mean_reward, "std_reward:", std_reward)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
print("Done")

# Ubuntu：use_gui=True,render_mode="rgb_array",vec_env.render("rgb_array") 在图形状态训练及评估正常，enjoy没有消息。
# 在terminal上去行时报错：AttributeError: type object 'gui' has no attribute 'DEFAULT_VIEW'

# Ubuntu：use_gui=True,render_mode="human",vec_env.render("human") ,在图形状态下运行时，启动了sumo-gui。
