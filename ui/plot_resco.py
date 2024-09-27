import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def plot_traffic_performance(csv_file):
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        return

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制每个指标的线图
    plt.plot(df['step'], df['system_total_stopped'], label='System Total Stopped')
    plt.plot(df['step'], df['system_total_waiting_time'], label='System Total Waiting Time')
    plt.plot(df['step'], df['system_mean_waiting_time'], label='System Mean Waiting Time')
    plt.plot(df['step'], df['system_mean_speed'], label='System Mean Speed')

    # 设置图表标题和轴标签
    plt.title('Traffic System Performance Indicators')
    plt.xlabel('Step')
    plt.ylabel('Value')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 生成输出文件名
    output_file = os.path.splitext(csv_file)[0] + '_performance.png'

    # 保存图形
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

    # 显示图形
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_resco.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    plot_traffic_performance(csv_file)
