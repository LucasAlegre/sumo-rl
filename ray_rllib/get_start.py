from pprint import pprint

import numpy as np
import torch
from ray import train, tune

from ray.rllib.algorithms.ppo import PPOConfig


def python_api():
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("CartPole-v1")
        .env_runners(num_env_runners=1)
    )

    algo = config.build()

    for i in range(10):
        result = algo.train()
        result.pop("config")
        pprint(result)

        if i % 5 == 0:
            checkpoint_dir = algo.save_to_path()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


def use_tune():
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("CartPole-v1")
        .training(
            lr=tune.grid_search([0.01, 0.001, 0.0001]),
        )
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            stop={"env_runners/episode_return_mean": 150.0},
        ),
    )

    tuner.fit()


def tune_result():
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("CartPole-v1")
        .training(
            lr=tune.grid_search([0.01, 0.001, 0.0001]),
        )
    )
    # Tuner.fit() allows setting a custom log directory (other than ~/ray-results).
    local_dir = "/Users/xnpeng/sumoptis/sumo-rl/ray_results"
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            stop={"num_env_steps_sampled_lifetime": 20000},
            checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
            storage_path=local_dir,
        ),
    )

    results = tuner.fit()

    # Get the best result based on a particular metric.
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean", mode="max"
    )

    print("best_result:\n")
    pprint(best_result)

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint

    print("\nbest_checkpoint:\n")
    pprint(best_checkpoint)


def load_restore_checkpoint():
    checkpoint_path = "/Users/xnpeng/sumoptis/sumo-rl/ray_results/PPO_2024-11-15_17-13-01/PPO_CartPole-v1_d108d_00000_0_lr=0.0100_2024-11-15_17-13-02/checkpoint_000000"
    from ray.rllib.algorithms.algorithm import Algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print("checkpoint_path:", checkpoint_path)
    print("checkpoint loaded", algo)
    print(algo.config)


def compute_action():
    import pathlib
    import gymnasium as gym
    import numpy as np
    import torch
    from ray.rllib.core.rl_module import RLModule

    env = gym.make("CartPole-v1")
    checkpoint_path = "/Users/xnpeng/sumoptis/sumo-rl/ray_results/PPO_2024-11-15_17-13-01/PPO_CartPole-v1_d108d_00000_0_lr=0.0100_2024-11-15_17-13-02/checkpoint_000000"

    # Create only the neural network (RLModule) from our checkpoint.
    rl_module = RLModule.from_checkpoint(
        pathlib.Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    )["default_policy"]

    episode_return = 0
    terminated = truncated = False

    obs, info = env.reset()

    while not terminated and not truncated:
        # Compute the next action from a batch (B=1) of observations.
        torch_obs_batch = torch.from_numpy(np.array([obs]))
        action_logits = rl_module.forward_inference({"obs": torch_obs_batch})[
            "action_dist_inputs"
        ]
        # The default RLModule used here produces action logits (from which
        # we'll have to sample an action or use the max-likelihood one).
        action = torch.argmax(action_logits[0]).numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

    print(f"Reached episode return of {episode_return}.")


def policy_state():
    algo = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("CartPole-v1")
        .env_runners(num_env_runners=1)
    ).build()

    # Get weights of the algo's RLModule.
    state = algo.get_module().get_state()
    print("algo.get_module().get_state():\n ", state)

    # Same as above
    algo.env_runner.module.get_state()

    # Get list of weights of each EnvRunner, including remote replicas.
    algo.env_runner_group.foreach_worker(lambda env_runner: env_runner.module.get_state())

    # Same as above, but with index.
    algo.env_runner_group.foreach_worker_with_id(
        lambda _id, env_runner: env_runner.module.get_state()
    )


def preprocess_observation():
    try:
        import gymnasium as gym

        env = gym.make("ale_py:ALE/Pong-v5")
        obs, infos = env.reset()
    except Exception:
        import gym

        env = gym.make("PongNoFrameskip-v4")
        obs = env.reset()

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations.
    from ray.rllib.models.preprocessors import get_preprocessor

    prep = get_preprocessor(env.observation_space)(env.observation_space)
    # <ray.rllib.models.preprocessors.GenericPixelPreprocessor object at 0x7fc4d049de80>

    # Observations should be preprocessed prior to feeding into a model
    print(obs)
    # (210, 160, 3)
    res = prep.transform(obs)
    print(res)
    # (84, 84, 3)


def policy_action():
    # Get a reference to the policy
    from ray.rllib.algorithms.dqn import DQNConfig

    algo = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .framework("torch")
        .environment("CartPole-v1")
        .env_runners(num_env_runners=0)
        .training(
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
            }
        )
    ).build()

    policy = algo.get_policy()
    # <ray.rllib.policy.eager_tf_policy.PPOTFPolicy_eager object at 0x7fd020165470>

    # Run a forward pass to get model output logits. Note that complex observations
    # must be preprocessed as in the above code block.
    logits, _ = policy.model({"obs": torch.from_numpy(np.array([[0.1, 0.2, 0.3, 0.4]]))})
    # (<tf.Tensor: id=1274, shape=(1, 2), dtype=float32, numpy=...>, [])

    # Compute action distribution given logits
    print(policy.dist_class)
    # <class_object 'ray.rllib.models.tf.tf_action_dist.Categorical'>
    dist = policy.dist_class(logits, policy.model)
    # <ray.rllib.models.tf.tf_action_dist.Categorical object at 0x7fd02301d710>

    # Query the distribution for samples, sample logps
    dist.sample()
    # <tf.Tensor: id=661, shape=(1,), dtype=int64, numpy=..>
    dist.logp(torch.tensor([1]))
    # <tf.Tensor: id=1298, shape=(1,), dtype=float32, numpy=...>

    # Get the estimated values for the most recent forward pass
    res = policy.model.value_function()
    # <tf.Tensor: id=670, shape=(1,), dtype=float32, numpy=...>
    print(res)

    print(policy.model)


def get_q_values():
    # Get a reference to the model through the policy
    import numpy as np
    import torch

    from ray.rllib.algorithms.dqn import DQNConfig

    algo = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .framework("torch")
        .environment("CartPole-v1")
        .training(
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
            }
        )
    ).build()
    model = algo.get_policy().model
    # <ray.rllib.models.catalog.FullyConnectedNetwork_as_DistributionalQModel ...>

    # List of all model variables
    list(model.parameters())

    # Run a forward pass to get base model output. Note that complex observations
    # must be preprocessed. An example of preprocessing is
    # examples/offline_rl/saving_experiences.py
    model_out = model({"obs": torch.from_numpy(np.array([[0.1, 0.2, 0.3, 0.4]]))})
    # (<tf.Tensor: id=832, shape=(1, 256), dtype=float32, numpy=...)
    print("\n=====================model_out:", model_out.shape())
    print(model_out)

    # Access the base Keras models (all default models have a base)
    print("\n=====================model")
    print(model)
    """
    Model: "model"
    _______________________________________________________________________
    Layer (type)                Output Shape    Param #  Connected to
    =======================================================================
    observations (InputLayer)   [(None, 4)]     0
    _______________________________________________________________________
    fc_1 (Dense)                (None, 256)     1280     observations[0][0]
    _______________________________________________________________________
    fc_out (Dense)              (None, 256)     65792    fc_1[0][0]
    _______________________________________________________________________
    value_out (Dense)           (None, 1)       257      fc_1[0][0]
    =======================================================================
    Total params: 67,329
    Trainable params: 67,329
    Non-trainable params: 0
    ______________________________________________________________________________
    """

    # Access the Q value model (specific to DQN)
    print("\n=====================model.get_q_value_distributions(model_out[0])[0]")
    print(model.get_q_value_distributions(model_out[0])[0])
    # tf.Tensor([[ 0.13023682 -0.36805138]], shape=(1, 2), dtype=float32)
    # ^ exact numbers may differ due to randomness

    print("\n=====================model.advantage_module")
    print(model.advantage_module)

    # Access the state value model (specific to DQN)
    print("\n=====================model.get_state_value(model_out[0])")
    print(model.get_state_value(model_out[0]))
    # tf.Tensor([[0.09381643]], shape=(1, 1), dtype=float32)
    # ^ exact number may differ due to randomness

    print("\n=====================model.value_module")
    print(model.value_module)


def algorithm_config():
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
    # Construct a generic config object, specifying values within different
    # sub-categories, e.g. "training".
    config = (PPOConfig().training(gamma=0.9, lr=0.01)
              .environment(env="CartPole-v1")
              .resources(num_gpus=0)
              .env_runners(num_env_runners=0)
              .callbacks(MemoryTrackingCallbacks)
              )
    # A config object can be used to construct the respective Algorithm.
    rllib_algo = config.build()

    from ray.rllib.algorithms.ppo import PPOConfig
    from ray import tune
    # In combination with a tune.grid_search:
    config = PPOConfig()
    config.training(lr=tune.grid_search([0.01, 0.001]))
    # Use `to_dict()` method to get the legacy plain python config dict
    # for usage with `tune.Tuner().fit()`.
    tune.Tuner("PPO", param_space=config.to_dict())


if __name__ == "__main__":
    # python_api()
    # use_tune()
    # tune_result()
    # load_restore_checkpoint()
    # compute_action()
    # policy_state()
    # preprocess_observation()
    # policy_action()
    # get_q_values()
    algorithm_config()
