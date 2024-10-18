import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def analyze_traffic_log(csv_file_path):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 创建一个2x3的子图布局
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('交通系统性能指标随时间变化', fontsize=16)

    # 绘制每个指标的曲线
    metrics = [
        ('system_total_stopped', '系统总停止车辆数', '停止车辆数'),
        ('system_total_waiting_time', '系统总等待时间', '等待时间'),
        ('system_mean_waiting_time', '系统平均等待时间', '平均等待时间'),
        ('system_mean_speed', '系统平均速度', '平均速度'),
        ('agents_total_stopped', '代理总停止车辆数', '停止车辆数'),
        ('agents_total_accumulated_waiting_time', '代理总累积等待时间', '累积等待时间')
    ]

    for i, (metric, title, ylabel) in enumerate(metrics):
        row, col = divmod(i, 3)
        axs[row, col].plot(df['step'], df[metric])
        axs[row, col].set_title(title)
        axs[row, col].set_xlabel('时间步')
        axs[row, col].set_ylabel(ylabel)

    # 调整子图布局
    plt.tight_layout()

    # 创建保存图片的目录
    output_dir = os.path.dirname(csv_file_path)
    output_file = os.path.join(output_dir, 'traffic_performance_metrics.png')

    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_file}")

    # 显示图表
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python log_analysis.py <csv文件路径>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 '{csv_file_path}' 不存在")
        sys.exit(1)
    
    analyze_traffic_log(csv_file_path)