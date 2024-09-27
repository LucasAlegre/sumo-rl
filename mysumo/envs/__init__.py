import gymnasium as gym

# 注册环境
gym.envs.register(
    id='SumoEnv-v0',  # 环境的唯一标识符，格式为 'YourEnvName-vX'
    entry_point='mysumo.envs.sumo_env:SumoEnv',  # 入口点，格式为 'module_name:ClassName'
    kwargs={"single_agent": True},
)
