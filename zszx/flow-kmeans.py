import pandas as pd
import numpy as np

# 从文件中读取数据
data = pd.read_csv('zszx/data/output_data.txt', sep='\t', header=None, names=['timestamp', 'direction', 'vehicle_id'])

# 转换时间戳为 datetime 格式
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 计算每小时流量
data['hour'] = data['timestamp'].dt.hour

# 计算每小时的流量，按小时和方向分组
hourly_flow = data.groupby(['hour', 'direction']).size().reset_index(name='flow_value')

# 计算每小时总流量
total_hourly_flow = hourly_flow.groupby('hour')['flow_value'].sum().reset_index()

# 计算流量变化率（每小时流量的差值）
total_hourly_flow['flow_change'] = total_hourly_flow['flow_value'].diff().abs()

# 设置流量变化阈值，变化较大的地方作为时段分割点
threshold = total_hourly_flow['flow_change'].quantile(0.75)  # 设置为流量变化的75%分位数

# 计算时间段的分割点
change_points = total_hourly_flow[total_hourly_flow['flow_change'] > threshold].index.tolist()

# 计算时段分割点的小时数
change_hours = total_hourly_flow.loc[change_points, 'hour'].values

# 根据change_hours进行时段划分
time_periods = {}
time_periods_names = ['night_peak', 'morning_peak', 'day_flat', 'evening_peak', 'evening_flat', 'night_flat']

# 起始时段，假设夜低峰从0点开始
last_end = 0
for idx, start_hour in enumerate(change_hours):
    time_periods[time_periods_names[idx]] = {'start': last_end, 'end': start_hour}
    last_end = start_hour

# 最后一段从最后一个分割点到24点
time_periods[time_periods_names[-1]] = {'start': last_end, 'end': 24}

# 打印时间段分割点
print("Time periods based on change points:")
for period, time_range in time_periods.items():
    print(f"{period}: {time_range['start']} - {time_range['end']}")

# 定义每个时段的车辆生成速率（车辆每秒生成数），按方向区分
time_periods_flow = {}

# 计算每个时段每小时各方向的流量，并计算生成速率
for period, time_range in time_periods.items():
    # 获取该时段的数据
    period_data = hourly_flow[(hourly_flow['hour'] >= time_range['start']) & (hourly_flow['hour'] < time_range['end'])]

    # 按方向分组并计算每个方向的流量总和
    direction_flow = period_data.groupby('direction')['flow_value'].sum().reset_index()

    # 计算每分钟生成的车辆数量 (流量 / 60)
    direction_flow['generation_rate'] = (direction_flow['flow_value'] / 60).round(2)

    # 存储到time_periods_flow字典中
    time_periods_flow[period] = {}
    for _, row in direction_flow.iterrows():
        direction = row['direction'].upper()  # 使用大写方向
        time_periods_flow[period][direction] = row['generation_rate']

# 输出结果
print("\nCalculated vehicle generation rates per time period:")
print(time_periods_flow)

"""
运行正确：

Time periods based on change points:
night_peak: 0 - 6
morning_peak: 6 - 7
day_flat: 7 - 17
evening_peak: 17 - 18
evening_flat: 18 - 19
night_flat: 19 - 24

Calculated vehicle generation rates per time period:
time_periods_flow = {
    'night_peak': {'ES': 1.12, 'EW': 4.42, 'NE': 0.48, 'NS': 7.95, 'SN': 7.13, 'SW': 5.67, 'WE': 7.57, 'WN': 1.53},
    'morning_peak': {'ES': 1.17, 'EW': 5.08, 'NE': 0.55, 'NS': 5.02, 'SN': 5.83, 'SW': 7.27, 'WE': 6.23, 'WN': 2.18},
    'day_flat': {'ES': 38.27, 'EW': 110.17, 'NE': 21.63, 'NS': 146.57, 'SN': 154.05, 'SW': 134.23, 'WE': 201.95, 'WN': 68.25},
    'evening_peak': {'ES': 10.18, 'EW': 8.93, 'NE': 2.87, 'NS': 19.87, 'SN': 10.63, 'SW': 11.2, 'WE': 39.72, 'WN': 6.28},
    'evening_flat': {'ES': 4.2, 'EW': 5.67, 'NE': 1.45, 'NS': 11.28, 'SN': 6.62, 'SW': 10.27, 'WE': 19.98, 'WN': 2.48},
    'night_flat': {'ES': 4.95, 'EW': 10.9, 'NE': 2.5, 'NS': 15.3, 'SN': 15.53, 'SW': 25.05, 'WE': 31.53, 'WN': 3.85}
}

"""