from ray import tune

# Configure.
from ray.rllib.algorithms.ppo import PPOConfig
config = PPOConfig().environment(env="CartPole-v1").training(train_batch_size=4000)

# Train via Ray Tune.
tune.run("PPO", config=config)

