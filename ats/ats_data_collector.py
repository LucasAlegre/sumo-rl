import time
from typing import Dict, List, Any
import random  # 仅用于模拟数据，实际使用时应删除

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AtsDataCollector:
    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self.lanes = self._initialize_lanes()
        self.last_update_time = time.time()

    def _initialize_lanes(self) -> List[str]:
        # 在实际应用中，这里应该从配置文件或数据库中读取车道信息
        return [f"{self.intersection_id}_lane_{i}" for i in range(1, 5)]

    def update(self):
        # 在实际应用中，这个方法应该触发所有传感器的数据更新
        current_time = time.time()
        if current_time - self.last_update_time >= 1:  # 每秒更新一次
            self._update_sensors()
            self.last_update_time = current_time

    def _update_sensors(self):
        # 在实际应用中，这个方法应该从实际的传感器读取数据
        # 这里我们只是模拟数据
        for lane in self.lanes:
            self._update_vehicle_count(lane)
            self._update_vehicle_speeds(lane)

    def _update_vehicle_count(self, lane: str):
        # 模拟更新车辆数量
        setattr(self, f"{lane}_vehicle_count", random.randint(0, 10))

    def _update_vehicle_speeds(self, lane: str):
        # 模拟更新车辆速度
        setattr(self, f"{lane}_vehicle_speeds", [random.uniform(0, 20) for _ in range(getattr(self, f"{lane}_vehicle_count"))])

    def get_lane_vehicle_count(self, lane: str) -> int:
        self.update()
        return getattr(self, f"{lane}_vehicle_count", 0)

    def get_lane_mean_speed(self, lane: str) -> float:
        self.update()
        speeds = getattr(self, f"{lane}_vehicle_speeds", [])
        return sum(speeds) / len(speeds) if speeds else 0

    def get_lanes_density(self) -> List[float]:
        self.update()
        # 假设每个车道长100米，每辆车占5米
        return [min(1, self.get_lane_vehicle_count(lane) * 5 / 100) for lane in self.lanes]

    def get_lanes_queue(self) -> List[float]:
        self.update()
        # 假设速度低于2m/s的车辆视为排队
        return [sum(1 for speed in getattr(self, f"{lane}_vehicle_speeds", []) if speed < 2) * 5 / 100 for lane in self.lanes]

    def get_total_queued(self) -> int:
        self.update()
        return sum(int(queue * 20) for queue in self.get_lanes_queue())  # 20 = 100m / 5m

    def get_average_speed(self) -> float:
        self.update()
        all_speeds = [speed for lane in self.lanes for speed in getattr(self, f"{lane}_vehicle_speeds", [])]
        return sum(all_speeds) / len(all_speeds) if all_speeds else 0

    def get_lane_queue(self, lane: str) -> float:
        self.update()
        lanes_queue = self.get_lanes_queue()
        lane_index = self.lanes.index(lane)
        return lanes_queue[lane_index]