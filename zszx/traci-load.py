import random

import traci
import time

# 定义每个时段的车辆生成速率（车辆每秒生成数），按方向区分
time_periods_flow = {
    'night_peak': {'ES': 1.12, 'EW': 4.42, 'NE': 0.48, 'NS': 7.95, 'SN': 7.13, 'SW': 5.67, 'WE': 7.57, 'WN': 1.53},
    'morning_peak': {'ES': 1.17, 'EW': 5.08, 'NE': 0.55, 'NS': 5.02, 'SN': 5.83, 'SW': 7.27, 'WE': 6.23, 'WN': 2.18},
    'day_flat': {'ES': 38.27, 'EW': 110.17, 'NE': 21.63, 'NS': 146.57, 'SN': 154.05, 'SW': 134.23, 'WE': 201.95, 'WN': 68.25},
    'evening_peak': {'ES': 10.18, 'EW': 8.93, 'NE': 2.87, 'NS': 19.87, 'SN': 10.63, 'SW': 11.2, 'WE': 39.72, 'WN': 6.28},
    'evening_flat': {'ES': 4.2, 'EW': 5.67, 'NE': 1.45, 'NS': 11.28, 'SN': 6.62, 'SW': 10.27, 'WE': 19.98, 'WN': 2.48},
    'night_flat': {'ES': 4.95, 'EW': 10.9, 'NE': 2.5, 'NS': 15.3, 'SN': 15.53, 'SW': 25.05, 'WE': 31.53, 'WN': 3.85}
}


# 初始化SUMO仿真
def start_sumo():
    # 启动SUMO仿真，加载初始需求文件
    traci.start(["sumo-gui", "-c", "zszx/net/zszx.sumocfg"])


counter = 0


# 动态控制流量，模拟时段变化
def adjust_traffic_flow(time_period, current_time):
    flows = time_periods_flow[time_period]
    global counter
    # 每60秒钟检查一次并生成车辆
    if current_time % 60 == 0:
        for direction, flow_rate in flows.items():
            # print(f"current_time {current_time}, direction {direction}, flow_rate {flow_rate}")
            num_vehicles_to_add = int(flow_rate)  # 每60秒生成的车辆数量
            # print("num_vehicles_to_add: ", num_vehicles_to_add)
            for _ in range(num_vehicles_to_add):
                # 随机选择车辆ID和路径
                route = direction.upper()  # 这里的 route 就是每个方向的 ID，如 'NS', 'NE' 等
                vehID = f"veh_{current_time}_{counter}"
                counter += 1
                depart = random.randint(1, 60) + current_time
                print(f"Added vehicle {vehID}, route {route}, depart {depart}")
                traci.vehicle.add(vehID=vehID, routeID=route, depart=depart, departLane="best", departSpeed="10")


# 运行仿真
def run_simulation():
    start_sumo()

    # 仿真循环，假设仿真运行24小时
    while True:
        current_time = traci.simulation.getTime()

        # 判断当前时段
        if 0 <= current_time / 3600 < 6:  # 夜低峰
            time_period = 'night_peak'
        elif 6 <= current_time / 3600 < 7:  # 早高峰
            time_period = 'morning_peak'
        elif 7 <= current_time / 3600 < 17:  # 日平峰
            time_period = 'day_flat'
        elif 17 <= current_time / 3600 < 18:  # 晚高峰
            time_period = 'evening_peak'
        elif 18 <= current_time / 3600 < 19:  # 晚高峰
            time_period = 'evening_flat'
        else:  # 晚平峰
            time_period = 'night_flat'

        # 调整流量（动态生成车辆）
        adjust_traffic_flow(time_period, current_time)

        # 进行下一步仿真
        traci.simulationStep()
        time.sleep(0.03)  # 控制仿真步骤的时间间隔

        if current_time > 100000:
            break

    traci.close()


# 启动仿真
run_simulation()

"""

在仿真过程中，按时间段按概率生成各个方向的车辆，通过traci加入系统，车流生成完美符合规律。

Time periods:
Time Period 1: 0:00 to 6:00
Time Period 2: 6:00 to 7:00
Time Period 3: 7:00 to 17:00
Time Period 4: 17:00 to 18:00
Time Period 5: 18:00 to 19:00
Time Period 6: 19:00 to 24:00
"""
