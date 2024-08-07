import sys

import numpy as np

sys.path.append('..')

from aits.RealWorldEnv import RealWorldEnv
from aits.TrainingManager import TrainingManager

def test_real_world_env(action_space_type='auto'):
    # 创建环境
    env = RealWorldEnv(
        intersection_ids=["intersection_1", "intersection_2"],
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=30,
        num_seconds=3600,
        reward_fn="queue",
        action_space_type = action_space_type
    )

    print(f"Action space type: {action_space_type}")
    print(f"Action space: {env.action_space}")

    # 重置环境
    observations = env.reset()
    print("Initial observations:", observations)

    # 运行几个步骤
    for i in range(10):
        print(f"\nStep {i + 1}")
        # 随机选择一个动作
        action = env.action_space.sample()
        print("Actions:", action)

        # 执行步骤
        observations, reward, terminated, truncated, info = env.step(action)
        print("Observations:", observations)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Info:", info)

        if terminated or truncated:
            print("Environment is done. Resetting.")
            observations = env.reset()

    # 关闭环境
    env.close()

def test_training_manager():
    config = {
        'env_params': {
            'intersection_ids': ["intersection_1", "intersection_2"],
            'delta_time': 5,
            'yellow_time': 2,
            'min_green': 5,
            'max_green': 30,
            'num_seconds': 3600,
            'reward_fn': "queue"
        },
        'algorithm': 'SAC',
        'total_timesteps': 10000,
        'algo_params': {
            'learning_rate': 1e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 64,
            'gamma': 0.99,
            'tau': 0.005,
        }
    }

    trainer = TrainingManager(config)
    trainer.train()
    trainer.evaluate()
    trainer.save_model()


if __name__ == "__main__":

    test_me = sys.argv[1]
    print("Testing {}".format(test_me))

    if test_me == "1":
        print("Testing environment with auto action space")
        test_real_world_env('auto')
        print("\nTesting environment with discrete action space")
    elif test_me == "2":
        print("Testing TrainingManager with SAC")
        test_training_manager()
        print("\nTesting TrainingManager with SAC")
    elif test_me == "3":
        print("Testing")