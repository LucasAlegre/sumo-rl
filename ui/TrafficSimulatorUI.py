import gradio as gr
from TrafficSimulator import TrafficSimulator, initialize_simulator, update_config, run_operation, save_model, load_model
from plot_figures import plot_process, plot_predict, plot_evaluation
import logging


def setup_logger():
    logger = logging.getLogger('TrafficSimulatorUI')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


def run_button_click(network_file, demand_file, algorithm, operation, total_timesteps, num_seconds):
    logger.info(f"network_file: {network_file.name}")
    logger.info(f"demand_file: {demand_file.name}")
    logger.info(f"algorithm: {algorithm}")
    logger.info(f"operation: {operation}")
    simulator = initialize_simulator(algorithm)
    update_config(num_seconds=num_seconds, total_timesteps=total_timesteps, net_file=network_file.name, route_file=demand_file.name,
                  algorithm=algorithm, operation=operation)
    for progress, output in run_operation(network_file.name, demand_file.name, operation):
        yield progress, output


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Advanced Intelligent Traffic Signal Control System")

    with gr.Row():
        with gr.Column(scale=2):
            network_file = gr.File(label="Network Model (.net.xml)", value="../mynets/net/my-intersection.net.xml", file_types=[".xml", ".net.xml"])
            demand_file = gr.File(label="Traffic Demand (.rou.xml)", value="../mynets/net/my-intersection-perhour.rou.xml",
                                  file_types=[".xml", ".rou.xml"])
        with gr.Column(scale=1):
            algorithm = gr.Dropdown(["DQN", "PPO", "A2C", "SAC"], value="PPO", label="Algorithm")
            operation = gr.Dropdown(["evaluate", "train", "predict"], value="evaluate", label="Operation")

    with gr.Row():
        total_timesteps = gr.Slider(1000, 1000000, value=100000, step=1000, label="Total Timesteps")
        num_seconds = gr.Slider(1000, 20000, value=3600, step=1000, label="Simulation Duration (seconds)")

    run_button = gr.Button("Start Simulation", variant="primary")
    progress = gr.Slider(0, 100, value=0, label="Progress", interactive=False)
    output = gr.Textbox(label="Output Information", lines=5)

    gr.Markdown("## Results Visualization")

    with gr.Row():
        with gr.Column(scale=2):
            train_process_file = gr.File(label="Select Training Process File", file_types=[".csv"])
            plot_train_button = gr.Button("Plot Training Process", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            predict_result_file = gr.File(label="Select Prediction Result File", file_types=[".json"])
            plot_predict_button = gr.Button("Plot Prediction Result", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            eval_result_file = gr.File(label="Select Evaluation File", file_types=[".txt"])
            plot_eval_button = gr.Button("Plot Evaluation Result", variant="secondary")

    plot_output = gr.Textbox(label="Plot Output", lines=2)
    plot_image = gr.Image(label="Generated Plot")

    # Binding event handling functions
    run_button.click(
        run_button_click,
        inputs=[network_file, demand_file, algorithm, operation, total_timesteps, num_seconds],
        outputs=[progress, output]
    )

    plot_train_button.click(
        plot_process,
        inputs=[train_process_file],
        outputs=[plot_image, plot_output])
    plot_predict_button.click(
        plot_predict,
        inputs=[predict_result_file],
        outputs=[plot_image, plot_output])
    plot_eval_button.click(
        plot_evaluation,
        inputs=[eval_result_file],
        outputs=[plot_image, plot_output])

demo.launch()
