import gradio as gr
from atscui.utils.visualization import Visualizer


class VisualizationTab:
    def __init__(self):
        self.visualizer = Visualizer()

    def render(self):
        with gr.Row():
            with gr.Column(scale=2):
                train_process_file = gr.File(
                    label="选择训练过程文件",
                    file_types=[".csv"]
                )
                plot_train_button = gr.Button(
                    "绘制训练过程图",
                    variant="secondary"
                )

        with gr.Row():
            with gr.Column(scale=2):
                eval_result_file = gr.File(
                    label="选择评估文件",
                    file_types=[".txt"]
                )
                plot_eval_button = gr.Button(
                    "绘制评估结果图",
                    variant="secondary"
                )

        plot_output = gr.Textbox(label="绘图输出", lines=2)
        plot_image = gr.Image(label="生成的图形")

        plot_train_button.click(
            self._plot_training_process,
            inputs=[train_process_file],
            outputs=[plot_image, plot_output]
        )

        plot_eval_button.click(
            self._plot_evaluation_results,
            inputs=[eval_result_file],
            outputs=[plot_image, plot_output]
        )

    def _plot_training_process(self, file):
        if file is None:
            return None, "请选择训练过程文件"

        try:
            fig = self.visualizer.plot_training_process(file.name)
            return fig, "训练过程图生成成功"
        except Exception as e:
            return None, f"绘图失败: {str(e)}"

    def _plot_evaluation_results(self, file):
        if file is None:
            return None, "请选择评估文件"

        try:
            fig = self.visualizer.plot_evaluation_results(file.name)
            return fig, "评估结果图生成成功"
        except Exception as e:
            return None, f"绘图失败: {str(e)}"