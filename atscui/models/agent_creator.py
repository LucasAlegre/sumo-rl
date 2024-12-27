from stable_baselines3 import DQN, A2C, PPO, SAC

from atscui.config import TrainingConfig
from atscui.models.agents import DQNAgent, A2CAgent, PPOAgent, SACAgent


def createAgent(env, config: TrainingConfig):
    if config.algo_name == "DQN":
        return DQNAgent(env, config)
    elif config.algo_name == "A2C":
        return A2CAgent(env, config)
    elif config.algo_name == "PPO":
        return PPOAgent(env, config)
    elif config.algo_name == "SAC":
        return SACAgent(env, config)
    else:
        raise ValueError("Algo_name {} not supported".format(config.algo_name))


def createAlgorithm(env, algo_name):
    if algo_name == "DQN":
        return DQN(env=env, policy="MlpPolicy")
    elif algo_name == "A2C":
        return A2C(env=env, policy="MlpPolicy")
    elif algo_name == "PPO":
        return PPO(env=env, policy="MlpPolicy")
    elif algo_name == "SAC":
        return SAC(env=env, policy="MlpPolicy")
    else:
        raise ValueError("Algo_name {} not supported".format(algo_name))
