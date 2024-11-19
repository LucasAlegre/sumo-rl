import pprint
from pathlib import Path

import ray
from ray.rllib.algorithms import SACConfig


def ppo_config():
    from ray.rllib.algorithms.ppo import PPOConfig

    config = PPOConfig()
    config.environment("CartPole-v1")
    config.env_runners(num_env_runners=1)
    config.training(
        gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size_per_learner=256
    )

    # Build a Algorithm object from the config and run 1 training iteration.
    algo = config.build()
    result = algo.train()
    print("result:\n", result)


def ppo_config_tune():
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray import air
    from ray import tune

    config = (
        PPOConfig()
        # Set the config object's env.
        .environment(env="CartPole-v1")
        # Update the config object's training parameters.
        .training(
            lr=0.001, clip_param=0.2
        )
    )

    result = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(stop={"training_iteration": 1}),
        param_space=config,
    ).fit()

    print("result:\n", result)


def dqn_config():
    from ray.rllib.algorithms.dqn.dqn import DQNConfig

    config = (
        DQNConfig()
        .environment("CartPole-v1")
        .training(replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 60000,
            "alpha": 0.5,
            "beta": 0.5,
        })
        .env_runners(num_env_runners=1)
    )
    algo = config.build()
    result = algo.train()
    print("result:\n", result)
    algo.stop()


def dqn_config_tune():
    from ray.rllib.algorithms.dqn.dqn import DQNConfig
    from ray import air
    from ray import tune

    config = (
        DQNConfig()
        .environment("CartPole-v1")
        .training(
            num_atoms=tune.grid_search([1, ])
        )
    )
    result = tune.Tuner(
        "DQN",
        run_config=air.RunConfig(stop={"training_iteration": 1}),
        param_space=config,
    ).fit()
    print("result:\n", result)


def sac_config():
    config = (
        SACConfig()
        .environment("Pendulum-v1")
        .env_runners(num_env_runners=1)
        .training(
            gamma=0.9,
            actor_lr=0.001,
            critic_lr=0.002,
            train_batch_size_per_learner=32,
        )
    )
    # Build the SAC algo object from the config and run 1 training iteration.
    algo = config.build()
    result = algo.train()
    print("result:\n", result)


def appo_config():
    from ray.rllib.algorithms.appo import APPOConfig
    config = APPOConfig().training(lr=0.01, grad_clip=30.0, train_batch_size=50)
    config = config.resources(num_gpus=0)
    config = config.env_runners(num_env_runners=1)
    config = config.environment("CartPole-v1")

    # Build an Algorithm object from the config and run 1 training iteration.
    algo = config.build()
    result = algo.train()
    print("result:\n", result)
    del algo


def appo_config_tune():
    from ray.rllib.algorithms.appo import APPOConfig
    from ray import air
    from ray import tune

    config = APPOConfig()
    # Update the config object.
    config = config.training(lr=tune.grid_search([0.001, ]))
    # Set the config object's env.
    config = config.environment(env="CartPole-v1")
    # Use to_dict() to get the old-style python config dict
    # when running with tune.
    result = tune.Tuner(
        "APPO",
        run_config=air.RunConfig(stop={"training_iteration": 1}, verbose=0),
        param_space=config.to_dict(),
    ).fit()
    print("result:\n", result)


def bc_config():
    from ray.rllib.algorithms.bc import BCConfig
    # Run this from the ray directory root.
    data_path = "ray_rllib/data/cartpole/large/large.json"
    base_path = Path(__file__).parents[1]
    input_path = base_path / data_path
    config = BCConfig().training(lr=0.00001, gamma=0.99).environment(env="CartPole-v1")
    config = config.offline_data(
        input_=[input_path.as_posix()],  # "./ray_rllib/data/cartpole/large",
    )

    # Build an Algorithm object from the config and run 1 training iteration.
    algo = config.build()
    result = algo.train()
    print("result:\n")
    pprint.pprint(result)


def bc_config_tune():
    from ray.rllib.algorithms.bc import BCConfig
    from ray import tune
    ray.init()

    config = BCConfig()
    # Print out some default values.
    print(config.beta)
    # Update the config object.
    config.training(
        lr=tune.grid_search([0.001, 0.0001]), beta=0.0
    )
    # Set the config object's data path.
    # Run this from the ray directory root.
    data_path = "ray_rllib/data/cartpole/large/large.json"
    base_path = Path(__file__).parents[1]
    input_path = base_path / data_path
    print("input_path:", input_path)
    config.offline_data(
        input_=[input_path.as_posix()],  # "./ray_rllib/data/cartpole/large",
        postprocess_inputs=True
    )
    # Set the config object's env, used for evaluation.
    config.environment(env="CartPole-v1")
    # Use to_dict() to get the old-style python config dict
    # when running with tune.
    result = tune.Tuner(
        "BC",
        param_space=config.to_dict(),
    ).fit()
    print("result:\n")
    pprint.pprint(result)


if __name__ == "__main__":
    # ppo_config()
    # ppo_config_tune()
    # dqn_config()
    # dqn_config_tune()
    # appo_config()
    # appo_config_tune()
    bc_config()
    # bc_config_tune()

"""
bc_config() 和 bc_config_tune() 运行出错：TypeError: argument of type 'PosixPath' is not iterable。
改成：input_=[input_path.as_posix()]，可以运行。
"""