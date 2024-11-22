import csv
import os
import sys
import traci
import sumolib
import time
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

# 定义全局变量
file_processed = False  # 用于标记文件是否已处理完
dt_start = 0


# 定义将时间字符串转换为仿真时刻的函数
def get_simulation_time(time_str):
    global dt_start
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")

    return (dt - dt_start).total_seconds()


# 逐行读取大文件并处理
def process_large_file(file_path):
    global file_processed
    global dt_start
    with open(file_path, 'r', encoding='utf-8') as file:
        counter = 0
        for line in file:
            # 解析每行数据
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue  # 跳过无效行

            time_str, direction, vehicle_id = parts
            if dt_start == 0:
                dt_start = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            sim_time = get_simulation_time(time_str)  # 转换为仿真时刻
            vehicle_id = vehicle_id + "_" + str(counter)
            counter += 1

            sumo_time = traci.simulation.getTime()

            # 在仿真中生成车辆
            if sim_time >= sumo_time:
                traci.vehicle.add(vehicle_id, routeID=direction, typeID="standard_car", depart=sim_time,
                                  departLane="random", departSpeed="15")  # 在指定时刻生成车辆
                print(f"生成 时间{sim_time} 车辆 {vehicle_id}，方向: {direction}")
    file_processed = True


# 运行仿真
def run_simulation(file_path):
    sumo_binary = sumolib.checkBinary('sumo-gui')
    traci.start([sumo_binary, "-c", "zszx/net/zszx.sumocfg"], label="default")
    traci.getConnection("default")

    lane_ids = traci.lane.getIDList()
    start_time = time.time()

    # 打开CSV文件保存结果
    with open('zszx/simu/queue_and_waiting_time.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'LaneID', 'QueueLength', 'AverageWaitingTime'])

        total_waiting_time = 0
        total_queue_length = 0
        tmp_waiting_time = 0
        # 设置仿真步骤
        while time.time() - start_time < 30 or traci.simulation.getMinExpectedNumber() > 0:
            # vehicle_ids = traci.vehicle.getIDList()
            # obj_number = traci.simulation.getMinExpectedNumber()
            # print(f"车辆: {len(vehicle_ids)}/{obj_number}: {vehicle_ids}")

            traci.simulationStep()  # 执行一步仿真
            current_time = traci.simulation.getTime()

            for lane_id in lane_ids:
                # 获取该车道上排队车辆的数量
                total_queue_length += traci.lane.getLastStepVehicleNumber(lane_id)

                # 获取该车道上车辆的平均等待时间
                # total_waiting_time = 0
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)

                for vehicle_id in vehicle_ids:
                    total_waiting_time += traci.vehicle.getWaitingTime(vehicle_id)

                if total_waiting_time != tmp_waiting_time:
                    tmp_waiting_time = total_waiting_time
                    # 保存到CSV
                    writer.writerow([current_time, total_queue_length, total_waiting_time])

            if not file_processed: process_large_file(file_path)  # 逐行处理历史流量数据
            time.sleep(0.00)  # 为了控制仿真速度，可以适当调整

    traci.close()  # 结束仿真


if __name__ == "__main__":
    file_path = "zszx/data/output_data.txt"  # 历史流量数据文件
    run_simulation(file_path)
