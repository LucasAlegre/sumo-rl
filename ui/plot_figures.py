import os
import json
import sys
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')


def replace_extension(pathname, new_extension):
    # 确保新扩展名以点开头
    if not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    # 分离路径和文件名
    directory, filename = os.path.split(pathname)
    # 分离文件名和扩展名
    name, _ = os.path.splitext(filename)
    # 组合新的文件名
    new_filename = name + new_extension
    # 组合新的完整路径
    new_pathname = os.path.join(directory, new_filename)
    return new_pathname


def plot_process(train_out_file, folder_name, file_name):
    process_fig = replace_extension(file_name, "png")
    output_path = os.path.join(folder_name, process_fig)
    # 加载数据
    df = pd.read_csv(train_out_file)

    # 数据预处理
    df['step'] = pd.to_numeric(df['step'])
    # 创建2x2的子图布局，调整整体图表大小
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # fig.suptitle('交通系统指标随时间变化\n{train_out_file}', fontsize=14)
    plt.suptitle(f'Traffic System Metrics Over Time\n{output_path}', fontsize=14)

    # 绘制系统平均速度
    axs[0, 0].plot(df['step'], df['system_mean_speed'], 'b-')
    axs[0, 0].set_title('Average Speed', fontsize=10)
    axs[0, 0].set_ylabel('speed', fontsize=8)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=6)
    axs[0, 0].grid(True)

    # 绘制系统停止的总车辆数
    axs[0, 1].plot(df['step'], df['system_total_stopped'], 'r-')
    axs[0, 1].set_title('System total stopped', fontsize=10)
    axs[0, 1].set_ylabel('vehicles', fontsize=8)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=6)
    axs[0, 1].grid(True)

    # 绘制系统总等待时间
    axs[1, 0].plot(df['step'], df['system_total_waiting_time'], 'g-')
    axs[1, 0].set_title('System total waiting time', fontsize=10)
    axs[1, 0].set_xlabel('timestep', fontsize=8)
    axs[1, 0].set_ylabel('waiting time', fontsize=8)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=6)
    axs[1, 0].grid(True)

    # 绘制代理总停止数
    axs[1, 1].plot(df['step'], df['agents_total_stopped'], 'm-')
    axs[1, 1].set_title('agents total stopped', fontsize=10)
    axs[1, 1].set_xlabel('timestep', fontsize=8)
    axs[1, 1].set_ylabel('agent number', fontsize=8)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=6)
    axs[1, 1].grid(True)

    # 调整子图间距
    plt.tight_layout()

    # 写入文件中
    plt.savefig(output_path)
    plt.close()
    print(f"图形已保存为{output_path}")
    return output_path


def plot_predict(predict_file, folder_name, file_name):
    predict_fig = replace_extension(file_name, "png")
    output_path = os.path.join(folder_name, predict_fig)

    # 读取JSON数据
    with open(predict_file, 'r') as file:
        data = json.load(file)

    # 将数据转换为DataFrame，不包括iteration
    df = pd.DataFrame(data)

    # 设置图形大小
    plt.figure(figsize=(20, 10))
    plt.suptitle(f'Traffic System Metrics Over Time\n{output_path}', fontsize=14)

    # 绘制各项指标的图形
    metrics = [
        'system_total_stopped', 'system_total_waiting_time', 'system_mean_waiting_time',
        'system_mean_speed', 'tl_1_stopped', 'tl_1_accumulated_waiting_time',
        'tl_1_average_speed', 'agents_total_stopped', 'agents_total_accumulated_waiting_time'
    ]

    # 计算需要的列数
    num_cols = (len(metrics) + 1) // 2  # 向上取整，确保有足够的列

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, num_cols, i)
        plt.plot(df['step'], df[metric], marker='o')
        plt.title(metric.replace('_', ' ').title(), fontsize=10)
        plt.xlabel('Step', fontsize=8)
        plt.ylabel('Value', fontsize=8)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.grid(True)

    # 调整子图布局
    plt.tight_layout()

    # 保存图形
    plt.savefig(output_path)
    plt.close()

    print(f"图形已保存为{output_path}")
    return output_path


def plot_evaluation(eval_folder, cross_name="my-intersection"):
    output_file = os.path.join(eval_folder, cross_name + "-eval.png")

    plt.figure(figsize=(12, 6))
    max_evaluations = 0

    for filename in os.listdir(eval_folder):
        if filename.startswith(cross_name + "-eval-") and filename.endswith(".txt"):
            file_path = os.path.join(eval_folder, filename)
            algorithm = filename.split('-')[-1].split('.')[0]  # 提取算法名称

            # 读取评估结果文件
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 解析数据
            mean_rewards = []
            std_rewards = []

            for line in lines:
                _, mean_reward, std_reward = line.strip().split(', ')
                mean_rewards.append(float(mean_reward))
                std_rewards.append(float(std_reward))

            # 使用评估次序作为x轴
            x = range(1, len(mean_rewards) + 1)
            max_evaluations = max(max_evaluations, len(mean_rewards))

            # 绘制数据
            plt.errorbar(x, mean_rewards, yerr=std_rewards, fmt='o-', capsize=5, label=algorithm)

    plt.title(f'Evaluation Results Comparison\n{output_file}')
    plt.xlabel('Evaluation Sequence')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # 设置x轴刻度为整数
    if max_evaluations > 0:
        plt.xticks(range(1, max_evaluations + 1))

    plt.tight_layout()

    # 设置x轴刻度为整数
    plt.xticks(range(1, max(plt.xticks()[0]) + 1))

    # 保存图形

    plt.savefig(output_file)
    plt.close()

    print(f"评估结果比较图形已保存为 {output_file}")

    return output_file
