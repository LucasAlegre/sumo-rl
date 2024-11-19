import pprint


def algo_config():
    # Configure.
    from ray.rllib.algorithms.ppo import PPOConfig
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("CartPole-v1")
        .training(train_batch_size_per_learner=4000)
    )

    # Build.
    algo = config.build()

    # Train.
    result = algo.train()
    result.pop("config")
    pprint.pprint(result)


def tune_config():
    from ray import tune

    # Configure.
    from ray.rllib.algorithms.ppo import PPOConfig
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("CartPole-v1")
        .training(train_batch_size_per_learner=4000)
    )

    # Train via Ray Tune.
    analysis = tune.run("PPO", config=config)
    pprint.pprint(analysis)


if __name__ == "__main__":
    # algo_config()
    tune_config()
