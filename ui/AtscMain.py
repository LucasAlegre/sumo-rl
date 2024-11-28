"""
用AtscMain代替AtscUI,进行模块化改造。
"""
import ntpath

import gradio as gr
import shlex
import os
import sys
from pathlib import Path
from plot_figures import plot_process, plot_predict, plot_evaluation
from stable_baselines3 import PPO, A2C, SAC
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from ui.utils import (add_directory_if_missing, extract_crossname_from_netfile,
                      write_eval_result, write_predict_result, get_relative_path,
                      extract_crossname_from_evalfile, get_gradio_file_info)

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.dqn import DQN
import mysumo.envs  # 确保自定义环境被注册

from mysumo.envs.sumo_env import SumoEnv, ContinuousSumoEnv


from ui.AtscUICreator import AtscUICreator


def setup_environment():
    """Setup necessary environment variables and paths"""
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

    # Add SUMO tools to Python path
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))


def main():
    setup_environment()

    ui = AtscUICreator()
    demo = ui.create_ui()
    demo.launch()


if __name__ == "__main__":
    main()
