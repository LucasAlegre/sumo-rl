import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()

config = PPOConfig().training(lr=tune.grid_search([0.01, 0.001, 0.0001]))

tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={"episode_reward_mean": 150},
    ),
    param_space=config,
)

tuner.fit()
