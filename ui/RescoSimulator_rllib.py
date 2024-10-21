import os
import sys
from pprint import pprint

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.tune.registry import register_env
import fire

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

p1 = os.path.abspath(__file__)
p2 = os.path.dirname(p1)
p3 = os.path.dirname(p2)
print("===========================")
print(f"p1 = {p1}, p2 = {p2}, p3 ={p3}")

sys.path.append('../')
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mysumo.envs.resco_envs import arterial4x4


def env_creator(config):
    env = arterial4x4(out_csv_name="outputs/grid4x4/arterial4x4", use_gui=False, yellow_time=2, fixed_ts=False)
    return env


register_env("arterial4x4", lambda config: ParallelPettingZooEnv(env_creator(config)))


def run(use_gui=False, episodes=50, load_model=False, save_model=True):
    ray.init()

    # 指定保存路径
    save_dir = os.path.join(os.getcwd(), "model_checkpoints")


    config = (
        DQNConfig()
        .environment("arterial4x4")
        .rollouts(num_env_runners=4)
        .training(
            train_batch_size=256,
            lr=1e-3,
            gamma=0.99,
        )
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    
    if load_model:
        # 加载已保存的检查点
        checkpoint_path = "path/to/checkpoint"
        algo = config.build()
        algo.restore(checkpoint_path)
    else:
        # 创建新的 DQN 训练器
        algo = config.build()

    for _ in range(episodes):
        result = algo.train()
        print(f"Episode reward mean: {result['env_runners']['episode_reward_mean']}")

    if save_model:
        checkpoint = algo.save()
        # print(f"Model saved at {checkpoint}")
        print("Saving checkpoint")
        # pprint(checkpoint)
        
    ray.shutdown()


def predict(use_gui=True, episodes=1):
    ray.init()

    config = (
        DQNConfig()
        .environment("arterial4x4")
        .framework("torch")
    )

    algo = config.build()

    # 加载已保存的检查点
    checkpoint_path = "path/to/checkpoint"
    algo.restore(checkpoint_path)

    env = PettingZooEnv(env_creator({}))

    for ep in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        print(f"Episode {ep + 1} reward: {episode_reward}")

    ray.shutdown()


if __name__ == "__main__":
    fire.Fire({
        'run': run,
        'predict': predict
    })

"""
程序不能运行，存在错误：
AttributeError: 'aec_to_parallel_wrapper' object has no attribute 'agent_selection'

解决办法请参考rllib_resco.py程序，PettingZooEnv -> ParallelPettingZooEnv

--> 找不到保存路径，如何设置保存路径？

"""