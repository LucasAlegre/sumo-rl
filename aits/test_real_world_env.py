import numpy as np
from aits.RealWorldEnv import RealWorldEnv


def test_real_world_env():
    # 创建环境
    env = RealWorldEnv(
        intersection_ids=["intersection_1", "intersection_2"],
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=30,
        num_seconds=3600,
        reward_fn="queue"
    )

    # 重置环境
    observations = env.reset()
    print("Initial observations:", observations)

    # 运行几个步骤
    for i in range(10):
        print(f"\nStep {i + 1}")

        # 为每个交叉口随机选择一个动作
        actions = {ts: env.action_spaces[ts].sample() for ts in env.intersection_ids}
        print("Actions:", actions)

        # 执行步骤
        observations, rewards, done, trunc, info = env.step(actions)

        print("Observations:", observations)
        print("Rewards:", rewards)
        print("Done:", done)
        print("Trunc:", trunc)
        print("Info:", info)

        if done:
            print("Environment is done. Resetting.")
            observations = env.reset()

    # 关闭环境
    env.close()


if __name__ == "__main__":
    test_real_world_env()
