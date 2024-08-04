import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


# 设置中文字体
# plt.rcParams['font.family'] = ['Heiti TC']  # 或者使用其他支持中文的字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


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


if (len(sys.argv) <= 1):
    print("usage: python plot_process.py train_out_file.csv")
    exit()

train_out_file = sys.argv[1]
# train_out_file = "out/my-intersection-algo-DQN_conn0_ep10.csv"
# 加载数据
df = pd.read_csv(train_out_file)

# 数据预处理
df['step'] = pd.to_numeric(df['step'])
# 创建2x2的子图布局，调整整体图表大小
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# fig.suptitle('交通系统指标随时间变化\n{train_out_file}', fontsize=14)
plt.suptitle(f'Traffic System Metrics Over Time\n{train_out_file}', fontsize=14)

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

# 显示图表
# plt.show()

# 写入文件中
process_fig = replace_extension(train_out_file,"png")
plt.savefig(process_fig)
plt.close()
print(f"图形已保存为{process_fig}")
