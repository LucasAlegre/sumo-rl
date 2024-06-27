import sys
from pathlib import Path

sys.path.append('..')

import mysumo.envs
from mysumo.envs.sumo_env import SumoEnv

print("hello world")

model_file = "model/my-intersection-modelPPO.zip"

file_path = Path(model_file)
if file_path.exists():
    print("load model=====加载训练模型==在原来基础上训练")
    # model.load(model_file)