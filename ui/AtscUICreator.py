from ui.AstcVisualTab import VisualizationTab
from ui.AtscTrainTab import TrainingTab
import gradio as gr


class AtscUICreator:
    def __init__(self):
        self.training_tab = TrainingTab()
        self.visualization_tab = VisualizationTab()

    def create_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 交通信号智能体训练系统")

            with gr.Tabs():
                with gr.TabItem("模型训练"):
                    self.training_tab.render()

                with gr.TabItem("结果可视化"):
                    self.visualization_tab.render()

        return demo
