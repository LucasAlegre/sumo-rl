import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 假设数据已经存储在 CSV 文件中
data = pd.read_csv('zszx/data/output_data.txt', sep='\t', header=None, names=['timestamp', 'direction', 'plate'])

# 将时间列转换为日期时间格式
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 提取出日期、小时和分钟（方便按时间段分组）
data['date'] = data['timestamp'].dt.date
data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute

# 按方向和日期统计流量
daily_flow_by_direction = data.groupby(['direction']).size()
daily_flow_by_direction = (daily_flow_by_direction / 48).astype(int)

# 按方向和小时统计流量
hourly_flow_by_direction = data.groupby(['direction', 'hour']).size().unstack(fill_value=0)

# 按方向和小时+分钟统计流量
minute_flow_by_direction = data.groupby(['direction', 'hour', 'minute']).size().unstack(fill_value=0)

# 按小时统计流量
plt.figure(figsize=(10, 6))
for direction in hourly_flow_by_direction.index:
    plt.plot(hourly_flow_by_direction.columns, hourly_flow_by_direction.loc[direction], marker='o', label=f'Flow Direction: {direction}')
plt.title('Traffic Flow by Hour and Direction')
plt.xlabel('Hour of Day')
plt.ylabel('Traffic Count')
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True)
plt.savefig('zszx/flow/hourly_flow_by_direction.png')
plt.show()

# 使用简单的趋势线（如线性回归）来观察流量趋势
from sklearn.linear_model import LinearRegression
import numpy as np

# 使用线性回归分析流量趋势
for direction in hourly_flow_by_direction.index:
    X = np.array(hourly_flow_by_direction.columns).reshape(-1, 1)
    y = hourly_flow_by_direction.loc[direction].values

    # 创建模型并拟合
    model = LinearRegression()
    model.fit(X, y)

    # 绘制趋势线
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_flow_by_direction.columns, hourly_flow_by_direction.loc[direction], marker='o', linestyle='-',
             label=f'Observed Flow ({direction})')
    plt.plot(hourly_flow_by_direction.columns, model.predict(X), linestyle='--', color='r', label=f'Trend Line ({direction})')
    plt.title(f'Traffic Flow Trend for Direction: {direction}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Traffic Count')
    plt.legend()
    plt.grid(True)
    # plt.show()

# 将按方向和日期统计的流量保存为 CSV 文件
daily_flow_by_direction.T.to_csv('zszx/flow/daily_flow_by_direction.csv', index=True)

# 将按方向和小时统计的流量保存为 CSV 文件
hourly_flow_by_direction.T.to_csv('zszx/flow/hourly_flow_by_direction.csv', header=True)

# 如果需要按方向和小时+分钟保存
minute_flow_by_direction.T.to_csv('zszx/flow/minute_flow_by_direction.csv', header=True)
