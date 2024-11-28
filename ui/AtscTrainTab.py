import json
import os
import re

import gradio as gr
from typing import Generator, Tuple

import pandas as pd
from matplotlib import pyplot as plt


class TrainingTab:
    def __init__(self):
        self.network_file = None
        self.demand_file = None
        self.progress = None
        self.render()

    def render(self):
        with gr.Row():
            with gr.Column(scale=2):
                network_file = gr.File(label="路网模型", value="mynets/net/my-intersection.net.xml", file_types=[".xml", ".net.xml"])
                demand_file = gr.File(label="交通需求", value="mynets/net/my-intersection-perhour.rou.xml", file_types=[".xml", ".rou.xml"])
            with gr.Column(scale=1):
                algorithm = gr.Dropdown(["DQN", "PPO", "A2C", "SAC"], value="DQN", label="算法模型")
                operation = gr.Dropdown(["EVAL", "TRAIN", "PREDICT", "ALL"], value="TRAIN", label="运行功能")

        with gr.Row():
            total_timesteps = gr.Slider(1000, 100000, value=100000, step=1000, label="训练步数")
            num_seconds = gr.Slider(1000, 20000, value=20000, step=1000, label="仿真秒数")

        run_button = gr.Button("开始运行", variant="primary")
        progress = gr.Slider(0, 100, value=0, label="进度", interactive=False)
        output = gr.Textbox(label="输出信息", lines=5)

    def run_training(self,
                     network_file,
                     demand_file,
                     algorithm,
                     operation,
                     total_timesteps,
                     num_seconds) -> Generator[Tuple[int, str], None, None]:
        # Training logic implementation
        pass


