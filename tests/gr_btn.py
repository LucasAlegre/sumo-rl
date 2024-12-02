import gradio as gr


def process_input(text, is_checked):
    if is_checked:
        return f"你输入了: {text}，并勾选了复选框。"
    else:
        return f"你输入了: {text}，但没有勾选复选框。"


with gr.Blocks() as demo:
    # 创建界面组件
    text_input = gr.Textbox(label="请输入文本")
    checkbox_input = gr.Checkbox(label="是否勾选")
    button_input = gr.Button("提交")

    # 启动界面
    button_input.click(fn=process_input,
                       inputs=[text_input, checkbox_input],
                       outputs=[text_input])
demo.launch()
