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
