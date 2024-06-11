from ray import air
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune import tune


def ray_sac_cartpole():
    config = SACConfig().training(gamma=0.9, lr=0.01)
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=4)
    print(config.to_dict())
    # Build a Algorithm object from the config and run 1 training iteration.
    algo = config.build(env="CartPole-v1")
    algo.train()


def ray_ppo_tune_cartpole():
    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]}, lr=0.01, gamma=0.9, kl_coeff=0.3)
        .rollouts(num_rollout_workers=4)
        .evaluation(evaluation_num_workers=1)
        .resources(num_gpus=0)
    )

    # tune.run(
    #     "PPO",
    #     run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
    #     param_space=config.to_dict(),
    # )


def ray_ppo_cartpole():
    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]}, lr=0.01, gamma=0.9, kl_coeff=0.3)
        .rollouts(num_rollout_workers=4)
        .evaluation(evaluation_num_workers=1)
        .resources(num_gpus=0)
    )

    algo = config.build()  # 2. build the algorithm,
    for _ in range(5):
        print(algo.train())  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.


def ray_ppo_taxi():
    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment("Taxi-v3")
        .rollouts(num_rollout_workers=2)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    algo = config.build()  # 2. build the algorithm,

    for _ in range(5):
        print(algo.train())  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.


def ray_dqn_taxi():
    config = (  # 1. Configure the algorithm,
        DQNConfig()
        .environment("Taxi-v3")
        .rollouts(num_rollout_workers=2)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    algo = config.build()  # 2. build the algorithm,

    for _ in range(5):
        print(algo.train())  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.


"""
If you want to get a quick preview of which algorithms and environments RLlib supports, click on the dropdowns below:
https://docs.ray.io/en/latest/rllib/index.html

强化学习的两个框架 ray-rllib 和 stable-baselines3 的对比：接口简洁，支持并行；ray有超参数优化的功能，stable-baselines3没有。

"""
if __name__ == "__main__":
    ray_sac_cartpole()
