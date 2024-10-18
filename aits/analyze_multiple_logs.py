import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_csv_safely(file_path):
    try:
        # 尝试使用默认设置读取CSV
        df = pd.read_csv(file_path)
        return df
    except pd.errors.ParserError:
        logging.warning(f"无法使用默认设置读取文件 {file_path}，尝试其他方法...")
        try:
            # 尝试使用更灵活的分隔符
            df = pd.read_csv(file_path, sep=None, engine='python')
            return df
        except Exception as e:
            logging.error(f"无法读取文件 {file_path}: {str(e)}")
            return None


def analyze_multiple_logs(folder_path):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取文件夹中所有的CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 定义要分析的指标
    metrics = [
        ('system_total_stopped', '系统总停止车辆数', '停止车辆数'),
        ('system_total_waiting_time', '系统总等待时间', '等待时间'),
        ('system_mean_waiting_time', '系统平均等待时间', '平均等待时间'),
        ('system_mean_speed', '系统平均速度', '平均速度'),
        ('agents_total_stopped', '代理总停止车辆数', '停止车辆数'),
        ('agents_total_accumulated_waiting_time', '代理总累积等待时间', '累积等待时间')
    ]

    # 为每个指标创建一个图表
    for metric, title, ylabel in metrics:
        plt.figure(figsize=(12, 6))
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            df = read_csv_safely(file_path)
            if df is not None and metric in df.columns:
                plt.plot(df['step'], df[metric], label=f'回合 {csv_file.split("_")[-1].split(".")[0]}')
            else:
                logging.warning(f"文件 {csv_file} 中没有找到指标 {metric}")

        plt.title(title)
        plt.xlabel('时间步')
        plt.ylabel(ylabel)
        plt.legend()

        # 保存图表
        output_file = os.path.join(folder_path, f'{metric}_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"{title}比较图已保存至: {output_file}")

        plt.close()  # 关闭当前图表,避免内存问题


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python analyze_multiple_logs.py <arterial4x4文件夹路径>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个有效的文件夹")
        sys.exit(1)

    analyze_multiple_logs(folder_path)
