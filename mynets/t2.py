import sys
from pathlib import Path

sys.path.append('..')

import mysumo.envs
from mysumo.envs.sumo_env import SumoEnv

print("hello world")

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

value = os.environ.get('KMP_DUPLICATE_LIB_OK')
print(value)

if value is True:
    print("true")
else:
    print("false")
