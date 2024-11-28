import ntpath
import os
import re

import gradio as gr

from ui.AtscPlot import Visualizer
from ui.utils import extract_crossname_from_evalfile


class VisualizationTab:
    def render(self):
        with gr.Row():
            with gr.Column(scale=2):
                train_process_file = gr.File(label="选择训练过程文件", file_types=[".csv"])
                plot_train_button = gr.Button("绘制训练过程图", variant="secondary")

        with gr.Row():
            with gr.Column(scale=2):
                predict_result_file = gr.File(label="选择预测结果文件", file_types=[".json"])
                plot_predict_button = gr.Button("绘制预测结果图", variant="secondary")

        with gr.Row():
            with gr.Column(scale=2):
                eval_result_file = gr.File(label="选择评估文件", file_types=[".txt"])
                plot_eval_button = gr.Button("绘制评估结果图", variant="secondary")

        plot_output = gr.Textbox(label="绘图输出", lines=2)
        plot_image = gr.Image(label="生成的图形")

        plot_train_button.click(
            self._plot_training_process,
            inputs=[train_process_file],
            outputs=[plot_image, plot_output]
        )

        plot_predict_button.click(
            self._plot_prediction_result,
            inputs=[predict_result_file],
            outputs=[plot_image, plot_output])

        plot_eval_button.click(
            self._plot_evaluation_results,
            inputs=[eval_result_file],
            outputs=[plot_image, plot_output]
        )

    def _plot_training_process(self, file):
        if file is None:
            return None, "请选择训练过程文件"
        folder_name, filename = self._get_gradio_file_info(file)
        output_path = Visualizer.plot_process(file.name, folder_name, filename)
        return output_path, f"训练过程图已生成：{output_path}"

    def _plot_prediction_result(self, file):
        if file is None:
            return "请选择预测结果文件"
        folder_name, filename = self._get_gradio_file_info(file)
        output_path = Visualizer.plot_predict(file.name, folder_name, filename)
        return output_path, f"预测结果图已生成：{output_path}"

    def _plot_evaluation_results(self, file):
        if file is None:
            return None, "请选择评估文件"
        folder_name, filename = self._get_gradio_file_info(file)
        eval_filename = ntpath.basename(filename)
        cross_name = extract_crossname_from_evalfile(eval_filename)  # 提取路口名称
        output_path = Visualizer.plot_evaluation(folder_name, cross_name)
        return output_path, f"预测结果图已生成：{output_path}"

    def _get_gradio_file_info(self, file: gr.File):
        if file is None:
            return None, None

        # 获取原始文件名
        filename = os.path.basename(file.name)

        conn_ep = r'_conn(\d+)_ep(\d+)'

        # 推断预期的文件夹
        if 'eval' in filename:
            inferred_folder = './evals'
        elif 'predict' in filename:
            inferred_folder = './predicts'
        elif re.search(conn_ep, filename):
            inferred_folder = './outs'
        else:
            inferred_folder = './'  # 默认为当前目录

        return inferred_folder, filename
