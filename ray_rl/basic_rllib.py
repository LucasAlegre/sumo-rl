# Configure.
from ray.rllib.algorithms.ppo import PPOConfig
config = (PPOConfig()
          .environment(env="CartPole-v1")
          .training(train_batch_size=4000)
          )

# Build.
algo = config.build()

# Train.
while True:
    print(algo.train())