import pandas as pd
import xml.etree.ElementTree as ET

# 假设你的历史数据已经存储在一个CSV文件中
data = pd.read_csv('./zszx/output_data.txt', sep='\t', header=None, names=['timestamp', 'direction', 'plate'])

with open("./zszx/zszx.rou.xml", "w") as routes:
    print(
        """<routes>
    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="30" sigma="0.5" />
    <route id="WN" edges="w_t t_n"/>
    <route id="WE" edges="w_t t_e"/>
    <route id="WS" edges="w_t t_s"/>
    
    <route id="NW" edges="n_t t_w"/>
    <route id="NE" edges="n_t t_e"/>
    <route id="NS" edges="n_t t_s"/>
    
    <route id="EW" edges="e_t t_w"/>
    <route id="EN" edges="e_t t_n"/>
    <route id="ES" edges="e_t t_s"/>
    
    <route id="SW" edges="s_t t_w"/>
    <route id="SN" edges="s_t t_n"/>
    <route id="SE" edges="s_t t_e"/>""",
        file=routes,
    )

    start_time = 0
    veh_counter = 0
    # 假设每条数据包含时间戳、方向和车牌
    for _, row in data.iterrows():
        direction = row['direction']  # 可以根据方向设置车道等信息
        vehicle_id = row['plate']  # 车牌号
        timestamp = pd.to_datetime(row['timestamp'])  # 时间戳
        vehicle_id = vehicle_id + "_" + str(veh_counter)
        veh_counter += 1

        if start_time == 0:
            start_time = timestamp

        depart_time = (timestamp - start_time).total_seconds()

        print(
            '    <vehicle id="%s" type="standard_car" route="%s" depart="%s" departLane="random" departSpeed="20" />'
            % (vehicle_id, direction, depart_time),
            file=routes,
        )

    print("</routes>", file=routes)

"""
利用历史流量数据，生成需求文件(zszx.rou.xml)，该需求只宜对历史过完程进行重现。
由于历史数据是电警录制的离开路口车辆数据，而不是车辆到达数据，不同用来作为流量进行仿真。
这批数据的价值在于反推一天的流量模式，而且不准确。
"""