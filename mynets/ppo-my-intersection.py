import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="mynets/my-intersection/my-intersection.net.xml",
        route_file="mynets/my-intersection/my-intersection.rou.xml",
        out_csv_name="mynets/out/my-intersection-ppo",
        single_agent=True,
        use_gui=False,
        num_seconds=10000,
    )

    model = PPO(env=env,
                policy="MlpPolicy",
                learning_rate=1e-3,
                tensorboard_log="./tensorboard/ppo-my-intersection",
                verbose=1)

    print("train model=====")
    model.learn(total_timesteps=10000)
    print("save model=====")
    model.save("mynets/model/my-intersection-ppo")
    print("load model=====")
    model.load("mynets/model/my-intersection-ppo")

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
        print("obs======\n", i, obs)
        print("\nreward=====\n", i, reward)
        print("\ndones=\n", i, dones)
        print("\ninfo=====\n", i, info)
        # print(obs, reward, dones, info)
        # env.render()
    env.close()
    del model

# 参考 experiments/dqn_big-intersection.py 程序
# 本程序运行到predict循环时，env.render()语句抛出异常。
# 在Ubuntu上和本机上，这两个程序(dqn-my-intersection.py, dqn_big-intersection.py)都在env.render()语句抛出异常。
