import os
import sys

import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="my-intersection/my-intersection.net.xml",
        route_file="my-intersection/my-intersection.rou.xml",
        out_csv_name="out/my-intersection-dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=10000,
    )

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        tensorboard_log="./tensorboard/dqn-my-intersection",
        verbose=1,
    )
    # print("train model=====")
    # model.learn(total_timesteps=10000)
    # print("save model=====")
    # model.save("mynets/model/my-intersection-dqn")
    print("load model=====")
    model.load("model/my-intersection-dqn")

    # 评测模型
    print("evaluate policy====")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(mean_reward, std_reward)

    print("predict====")
    env = model.get_env()
    obs = env.reset()
    for i in range(10):
        action, state = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        print("\n======", i, "======")
        print("obs:", i, obs)
        print(" reward:", reward)
        print(" dones:", dones)
        print(" info:", info)
        # print(obs, reward, dones, info)
        # env.render()
    env.close()
    del model
# 参考 experiments/dqn_big-intersection.py 程序
# 本程序运行到predict循环时，env.render()语句抛出异常。
# 在Ubuntu上和本机上，这两个程序(dqn-my-intersection.py, dqn_big-intersection.py)都在env.render()语句抛出异常。
