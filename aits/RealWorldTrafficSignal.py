from typing import List, Callable, Union
import time
import numpy as np
from gymnasium import spaces

from aits.RealWorldDataCollector import RealWorldDataCollector
from aits.SignalController import SignalController


class RealWorldTrafficSignal:
    def __init__(
            self,
            env,
            ts_id: str,
            delta_time: int,
            yellow_time: int,
            min_green: int,
            max_green: int,
            begin_time: int,
            reward_fn: Union[str, Callable],
            data_collector: RealWorldDataCollector
    ):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.begin_time = begin_time
        self.reward_fn = reward_fn
        self.data_collector = data_collector

        self.signal_controller = SignalController(ts_id)

        self.last_measure = 0.0
        self.last_reward = None

        self._build_phases()

        self.lanes = self.signal_controller.get_controlled_lanes()
        self.out_lanes = [link[1] for link in self.signal_controller.get_controlled_links()]
        self.lanes_length = {lane: 100 for lane in self.lanes + self.out_lanes}  # 假设所有车道长度为100米

        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _build_phases(self):
        self.all_phases = self.signal_controller.phases
        self.green_phases = [phase for phase in self.all_phases if 'y' not in phase]
        self.num_green_phases = len(self.green_phases)

    def _build_observation_space(self):
        return spaces.Box(
            low=np.zeros(self.num_green_phases + 1 + 2 * len(self.lanes), dtype=np.float32),
            high=np.ones(self.num_green_phases + 1 + 2 * len(self.lanes), dtype=np.float32),
        )

    def _build_action_space(self):
        return spaces.Discrete(self.num_green_phases)

    def set_next_phase(self, new_phase: int):
        new_phase = int(new_phase)
        current_phase = self.signal_controller.get_current_phase()
        time_since_last_change = self.signal_controller.get_time_since_last_change()

        if current_phase == new_phase or time_since_last_change < self.yellow_time + self.min_green:
            self.signal_controller.set_phase(current_phase)
        else:
            yellow_phase = self.all_phases.index(self.all_phases[current_phase].replace('G', 'y'))
            self.signal_controller.set_phase(yellow_phase)
            time.sleep(self.yellow_time)
            self.signal_controller.set_phase(new_phase)

    def update(self):
        # 在实际应用中，这个方法可能需要进行一些状态更新
        pass

    def compute_observation(self):
        phase_id = [1 if self.signal_controller.get_current_phase() == i else 0 for i in range(self.num_green_phases)]
        min_green = [0 if self.signal_controller.get_time_since_last_change() < self.min_green + self.yellow_time else 1]
        density = self.data_collector.get_lanes_density()
        queue = self.data_collector.get_lanes_queue()
        return np.array(phase_id + min_green + density + queue, dtype=np.float32)

    def compute_reward(self):
        if callable(self.reward_fn):
            self.last_reward = self.reward_fn(self)
        elif self.reward_fn == "queue":
            self.last_reward = -self.data_collector.get_total_queued()
        elif self.reward_fn == "wait":
            ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
            reward = self.last_measure - ts_wait
            self.last_measure = ts_wait
            self.last_reward = reward
        else:
            raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")
        return self.last_reward

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        # 在实际应用中，这个方法应该从数据收集器获取数据
        # return [self.data_collector.get_lane_waiting_time(lane) for lane in self.lanes]
        # 使用队列长度作为等待时间的近似
        queues = self.data_collector.get_lanes_queue()
        # 假设每个排队的车辆等待时间为10秒
        return [queue * 10 for queue in queues]

    def get_average_speed(self) -> float:
        return self.data_collector.get_average_speed()

    def get_pressure(self):
        return sum(self.data_collector.get_lane_vehicle_count(lane) for lane in self.out_lanes) - \
            sum(self.data_collector.get_lane_vehicle_count(lane) for lane in self.lanes)

    def get_lanes_density(self) -> List[float]:
        return self.data_collector.get_lanes_density()

    def get_lanes_queue(self) -> List[float]:
        return self.data_collector.get_lanes_queue()

    def get_total_queued(self) -> int:
        return self.data_collector.get_total_queued()

    @property
    def time_to_act(self):
        return time.time() >= self.signal_controller.last_change_time + self.delta_time
