import gradio as gr
from atscui.ui.components.training_tab import TrainingTab
from atscui.ui.components.visualization_tab import VisualizationTab


class ATSCUI:

    def create_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 交通信号智能体训练系统")

            with gr.Tabs():
                with gr.TabItem("模型训练"):
                    self.training_tab = TrainingTab()
                    self.training_tab.render()

                with gr.TabItem("结果可视化"):
                    self.visualization_tab = VisualizationTab()
                    self.visualization_tab.render()

        return demo
