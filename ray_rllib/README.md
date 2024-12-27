## 说明

ray_rllib, ray_train, ray_tune三个模块是相应于ray/rllib, ray/train, ray/rune3个模块
的学习指南。

## ray_rllib

### get_start.py

- python_api()
- use_tune()
- tune_result()
- load_restore_checkpoint()
- compute_action()
- policy_state()
- preprocess_observation()
- policy_action()
- get_q_values()
- algorithm_config()

### rl_algorithmns.py

- ppo_config()
- ppo_config_tune()
- dqn_config()
- dqn_config_tune()
- appo_config()
- appo_config_tune()
- bc_config()
- bc_config_tune()

### h_trainning.py

分层强化学习。
- 1. 扁平(Flat)模式：单个智能体直接学习动作
- 2. 分层(Hierarchical)模式：通过高层和低层智能体协作来学习

### cartpole_server.py / cartpole_client.py

客户端/服务器模式

### BC行为克隆算法

BC算法是一种模仿学习的方法，其主要目的是让AI直接模仿和复制已有的示范行为。
