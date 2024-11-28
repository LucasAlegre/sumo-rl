import time
from datetime import datetime

import traci

dt_start = 0
counter = 0


# 逐批加载流量数据的函数，改进版本，避免每次打开文件
def load_traffic_data_batch(file, batch_size=50):
    """
    逐批读取历史流量数据。
    :param file: 已经打开的文件对象
    :param batch_size: 每次加载的记录数
    :return: 返回一个包含batch_size条记录的批次数据
    """
    global dt_start
    global counter
    for i in range(batch_size):
        line = file.readline()
        if line:
            time_str, direction, vehicle_id = line.strip().split('\t')
            if dt_start == 0:
                dt_start = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
            sim_time = (dt - dt_start).total_seconds()
            vehicle_id = vehicle_id + "_" + str(counter)
            sumo_time = traci.simulation.getTime()

            # 在仿真中生成车辆
            if sim_time >= sumo_time:
                traci.vehicle.add(vehicle_id, routeID=direction, typeID="standard_car", depart=sim_time,
                                  departLane="random", departSpeed="15")  # 在指定时刻生成车辆
                if i == batch_size - 1:
                    print(f"生成 时间{sim_time}, 车辆 {vehicle_id}，方向 {direction}")
            counter += 1
        else:
            break


# 启动仿真并逐步加载流量数据
def run_simulation_with_dynamic_traffic(file_path, sumo_binary="sumo-gui",
                                        sumo_config_file="zszx/net/zszx.sumocfg",
                                        batch_size=50,
                                        steps_per_batch=100):
    """
    运行SUMO仿真，在仿真过程中逐步加载历史流量数据。
    :param file_path: 流量数据文件路径
    :param sumo_binary: SUMO二进制可执行文件路径
    :param sumo_config_file: SUMO配置文件路径
    :param batch_size: 每次读取流量数据的批次大小
    :param steps_per_batch: 每次仿真运行多少步加载一次数据
    """
    # 连接TraCI
    traci.start([sumo_binary, "-c", sumo_config_file], label="default")
    traci.getConnection("default")

    # 初始化仿真步
    current_step = 0
    total_waiting_time = 0
    total_queue_length = 0
    tmp_waiting_time = 0
    lane_ids = traci.lane.getIDList()

    # 打开文件并保留文件对象
    with open(file_path, 'r', encoding='utf-8') as file:
        load_traffic_data_batch(file, batch_size)

        while traci.simulation.getMinExpectedNumber() > 0:
            current_step += 1
            # 在仿真循环中，每经过一定步数后，加载一个新的批次的流量数据
            if current_step % steps_per_batch == 0:
                print("current_step:", current_step)
                print(f"排队长度：{total_queue_length}，等待时间：{total_waiting_time}")
                # 更新数据索引，读取下一批流量数据
                load_traffic_data_batch(file, batch_size)
            # 运行仿真一步
            traci.simulationStep()

            for lane_id in lane_ids:
                current_time = traci.simulation.getTime()
                # 获取该车道上排队车辆的数量
                total_queue_length += traci.lane.getLastStepVehicleNumber(lane_id)
                # 获取该车道上车辆的平均等待时间
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)

                if current_step % steps_per_batch == 0 and len(vehicle_ids) > 0:
                    print("vehicle_ids:", vehicle_ids)

                for vehicle_id in vehicle_ids:
                    total_waiting_time += traci.vehicle.getWaitingTime(vehicle_id)

                if total_waiting_time != tmp_waiting_time:
                    tmp_waiting_time = total_waiting_time
    # 结束仿真
    traci.close()


if __name__ == "__main__":
    file_path = "zszx/data/output_data.txt"  # 历史流量数据文件
    run_simulation_with_dynamic_traffic(file_path)

"""
这个程序是利用历史数据(用generate_route_file.py生成)仿真信控过程。流量是历史数据，配时方案是真实的固定配时。

在仿真过程中分批次加载历史数据进入仿真系统，而不是一次性加载数据造成内存耗尽的情况。
比如，运行100个仿真步加载一批50条数据。注意，100仿真远小于100秒，所以加载的历史数据的发车时间(depart)不能小于100仿真步所需要的时间
（具体时间跟硬件相关），否则这些车辆超过了发车时间而被抛弃。
"""
