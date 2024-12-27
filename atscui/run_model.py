import os
import sys
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atscui.config.base_config import RunningConfig
from atscui.environment import createEnv
from atscui.models.agent_creator import createAlgorithm


def parse_config():
    return RunningConfig(
        net_file="/Users/xnpeng/sumoptis/sumo-rl/zszx/net/zszx-2.net.xml",
        rou_file="/Users/xnpeng/sumoptis/sumo-rl/zszx/net/zszx-perhour-3.rou.xml",
        model_path="/Users/xnpeng/sumoptis/sumo-rl/models/zszx-2-model-PPO.zip",
        csv_path="/Users/xnpeng/sumoptis/sumo-rl/outs",
        predict_path="/Users/xnpeng/sumoptis/sumo-rl/predicts",
        eval_path="/Users/xnpeng/sumoptis/sumo-rl/evals",
        single_agent=True,
        algo_name="PPO")


def get_obs(seed: int):
    observation_space = Box(low=0, high=1, shape=(43,), seed=seed, dtype=np.float32)

    # 生成符合要求的 observation (0,1) 之间的随机值
    observation = observation_space.sample()
    return observation


def running():
    config = parse_config()

    env = createEnv(config)
    model = createAlgorithm(env, config.algo_name)

    model_obj = Path(config.model_path)
    if model_obj.exists():
        print("==========load model==========")
        model.load(model_obj)

    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs)  # 通过状态变量预测动作变量
        print(obs)
        for _ in range(3):
            obs, reward, done, info = env.step(action)  # 来自探测器的状态变量
            print(action)


if __name__ == "__main__":
    running()

"""
加载训练过的算法模型model，根据环境env的状态obs选择动作action，让环境执行该动作。
该程序与图形界面里的PREDICT操作相同。

测试结果符合预期，即相同的状态observation，会产生相同的动作action。

=========================

注：黄灯不算相位。下表只有4个相位：0，1，2，3

<phase duration="40" state="GGGGGGrrrrrrrrrrrrrGGGGGGrrrrrrrrrrrrrr"/>
<phase duration="3"  state="yyyyyyrrrrrrrrrrrrryyyyyyrrrrrrrrrrrrrr"/>
<phase duration="20" state="rrrrrrGGGGrrrrrrrrrrrrrrrGGGGrrrrrrrrrr"/>
<phase duration="3"  state="rrrrrryyyyrrrrrrrrrrrrrrryyyyrrrrrrrrrr"/>
<phase duration="50" state="rrrrrrrrrrGGGGGGrrrrrrrrrrrrrGGGGGGrrrr"/>
<phase duration="5"  state="rrrrrrrrrryyyyyyrrrrrrrrrrrrryyyyyyrrrr"/>
<phase duration="20" state="rrrrrrrrrrrrrrrrGGGrrrrrrrrrrrrrrrrGGGG"/>
<phase duration="5"  state="rrrrrrrrrrrrrrrryyyrrrrrrrrrrrrrrrryyyy"/>

结果：
长向量是observation,shapge(43,)
短向量是action,shape(1,)

[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[1]
[1]
[1]
[[1.         0.         0.         0.         1.         0.
  0.04093887 0.         0.04093887 0.         0.02709538 0.02709538
  0.         0.02709538 0.         0.04242082 0.         0.04242082
  0.04242082 0.02741228 0.02741228 0.02741228 0.02741228 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]]
[2]
[2]
[2]
[[0.         0.         1.         0.         1.         0.08187773
  0.04093887 0.04093887 0.04093887 0.         0.05419075 0.
  0.         0.05419075 0.04242082 0.08484163 0.         0.12726244
  0.08484163 0.05482456 0.05482456 0.02741228 0.02741228 0.02741228
  0.04093887 0.         0.         0.04093887 0.         0.
  0.         0.         0.02709538 0.04242082 0.         0.
  0.04242082 0.04242082 0.         0.         0.         0.02741228
  0.        ]]
[3]
[3]
[3]
[[0.         0.         0.         1.         1.         0.08187773
  0.08187773 0.08187773 0.04093887 0.         0.02709538 0.05419075
  0.02709538 0.02709538 0.08484163 0.04242082 0.08484163 0.16968326
  0.12726244 0.05482456 0.08223684 0.05482456 0.02741228 0.
  0.04093887 0.04093887 0.04093887 0.04093887 0.         0.02709538
  0.         0.         0.         0.04242082 0.04242082 0.04242082
  0.12726244 0.08484163 0.02741228 0.02741228 0.         0.
  0.        ]]
[1]
[1]
[1]
[[0.         1.         0.         0.         1.         0.12281659
  0.12281659 0.08187773 0.04093887 0.         0.05419075 0.02709538
  0.05419075 0.05419075 0.08484163 0.08484163 0.12726244 0.04242082
  0.04242082 0.08223684 0.10964912 0.08223684 0.02741228 0.02741228
  0.08187773 0.08187773 0.04093887 0.         0.         0.02709538
  0.02709538 0.02709538 0.02709538 0.08484163 0.04242082 0.08484163
  0.         0.         0.05482456 0.05482456 0.02741228 0.02741228
  0.        ]]
[2]
[2]
[2]
[[0.         0.         1.         0.         1.         0.16375546
  0.12281659 0.12281659 0.04093887 0.         0.02709538 0.02709538
  0.         0.05419075 0.12726244 0.12726244 0.12726244 0.08484163
  0.08484163 0.02741228 0.05482456 0.02741228 0.05482456 0.02741228
  0.12281659 0.12281659 0.08187773 0.04093887 0.         0.
  0.         0.         0.02709538 0.08484163 0.08484163 0.12726244
  0.04242082 0.04242082 0.         0.         0.         0.02741228
  0.        ]]
[3]
[3]
[3]
[[0.         0.         0.         1.         1.         0.16375546
  0.16375546 0.16375546 0.04093887 0.         0.02709538 0.05419075
  0.         0.02709538 0.16968326 0.12726244 0.16968326 0.12726244
  0.12726244 0.05482456 0.08223684 0.05482456 0.02741228 0.
  0.16375546 0.12281659 0.12281659 0.04093887 0.         0.02709538
  0.         0.         0.         0.12726244 0.12726244 0.12726244
  0.08484163 0.08484163 0.02741228 0.02741228 0.02741228 0.
  0.        ]]
[0]
Step #100.00 (0ms ?*RT. ?UPS, TraCI: 14ms, vehicles TOT 86 ACT 52 BUF 0)                   
 Retrying in 1 seconds
[0]
[0]
[[1.         0.         0.         0.         0.         0.04093887
  0.         0.         0.04093887 0.         0.02709538 0.
  0.         0.02709538 0.04242082 0.         0.         0.04242082
  0.         0.02741228 0.         0.         0.02741228 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]]
[0]
[0]
[0]
[[1.         0.         0.         0.         1.         0.04093887
  0.04093887 0.         0.04093887 0.         0.05419075 0.
  0.         0.02709538 0.04242082 0.04242082 0.         0.08484163
  0.04242082 0.05482456 0.05482456 0.02741228 0.02741228 0.02741228
  0.         0.         0.         0.04093887 0.         0.
  0.         0.         0.         0.         0.         0.
  0.04242082 0.         0.         0.         0.         0.
  0.        ]]
[3]
[3]
[3]
[[0.         0.         0.         1.         1.         0.08187773
  0.04093887 0.04093887 0.04093887 0.         0.05419075 0.02709538
  0.02709538 0.02709538 0.04242082 0.04242082 0.04242082 0.12726244
  0.08484163 0.08223684 0.08223684 0.05482456 0.02741228 0.
  0.04093887 0.04093887 0.         0.04093887 0.         0.02709538
  0.         0.         0.         0.04242082 0.         0.
  0.08484163 0.04242082 0.02741228 0.02741228 0.02741228 0.
  0.        ]]
[3]
[3]
[3]

"""
