import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def plot_traffic_performance(csv_file):
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        return

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 创建2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Traffic System Performance Indicators')

    # 绘制每个指标的子图
    axs[0, 0].plot(df['step'], df['system_total_stopped'])
    axs[0, 0].set_title('System Total Stopped')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].grid(True)

    axs[0, 1].plot(df['step'], df['system_total_waiting_time'])
    axs[0, 1].set_title('System Total Waiting Time')
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].grid(True)

    axs[1, 0].plot(df['step'], df['system_mean_waiting_time'])
    axs[1, 0].set_title('System Mean Waiting Time')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].grid(True)

    axs[1, 1].plot(df['step'], df['system_mean_speed'])
    axs[1, 1].set_title('System Mean Speed')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].grid(True)

    # 调整子图布局
    plt.tight_layout()

    # 生成输出文件名
    output_file = os.path.splitext(csv_file)[0] + 'multi_performance.png'

    # 保存图形
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

    # 显示图形
    plt.show()

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Plot traffic performance indicators from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用绘图函数
    plot_traffic_performance(args.csv_file)
