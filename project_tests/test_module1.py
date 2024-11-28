import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from project_name import module1, module2, config, utils

print("test_module1")
module1.module1_func()
module2.module2_func()
config.get_config()
utils.get_utils()
