from time import sleep

import gradio as gr


class TrainingTab:
    def __init__(self):
        self.name = "Gradio App"
        self.control = None
        self.progress = None
        self.run_button = None
        self.progress = None

    def create_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 交通信号智能体训练系统")
            with gr.Row():
                with gr.Column(scale=2):
                    network_file = gr.File(label="路网模型", value="zszx/net/zszx.net.xml", file_types=[".xml", ".net.xml"])
                    demand_file = gr.File(label="交通需求", value="zszx/net/zszx-perhour-1.rou.xml", file_types=[".xml", ".rou.xml"])

            self.run_button = gr.Button("开始运行", variant="primary")
            self.progress = gr.Slider(0, 10000, value=0, label="进度", interactive=False)
            output = gr.Textbox(label="输出信息", lines=5)
            self.checkbox1 = gr.Checkbox(label="GUI", value=False)

            self.run_button.click(
                self.run_training,
                inputs=[network_file, demand_file, self.checkbox1],
                outputs=[self.progress, output]
            )
        return demo

    def run_training(self,
                     network_file,
                     demand_file
                     ):

        gr.update(elem_id=self.run_button.elem_id, interactive=False)

        if not network_file or not demand_file:
            yield 0, "请上传路网模型和交通需求文件"

        # 打印复选框的值
        print("checkbox1:", self.checkbox1.value)

        progress = 0
        for i in range(10000):
            sleep(0.1)
            progress += 1
            output = "progress: " + str(round(progress / 10000, 2)) + "%"
            gr.update(elem_id=self.progress.elem_id, value=progress)  # 更新进度条
            # yield progress, output

        gr.update(elem_id=self.run_button.elem_id, interactive=True)


if __name__ == "__main__":
    t = TrainingTab()
    app = t.create_ui()
    app.launch()
