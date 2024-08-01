import json
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
    print("usage: python plot_predict.py predict_file.json")
    exit()

predict_file = sys.argv[1]
# predict_file = "./predict/my-intersection-modelDQN.json"
# 读取JSON数据
with open(predict_file, 'r') as file:
    data = json.load(file)

# 将数据转换为DataFrame，不包括iteration
df = pd.DataFrame(data)

# 设置图形大小
plt.figure(figsize=(20, 10))
plt.suptitle(f'Traffic System Metrics Over Time\n{predict_file}', fontsize=14)

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
# plt.savefig('./predict/metrics_predict_plot.png')
predict_fig = replace_extension(predict_file, "png")
plt.savefig(predict_fig)
plt.close()

print(f"图形已保存为{predict_fig}")
