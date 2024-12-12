import time
from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Phase:
    def __init__(self, duration, state, minDur, maxDur):
        self.duration = duration  # 相位持续时间
        self.state = state  # 对应的交通信号状态
        self.minDur = minDur
        self.maxDur = maxDur

    def __repr__(self):
        return f"Phase(duration={self.duration}, state='{self.state}')"


def load_phases_from_file(file_path):
    """
    从XML文件加载交通信号相位信息，并返回一个Phase对象的列表。
    :param file_path: XML文件的路径
    :return: List[Phase] 对象的列表
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(file_path)
    root = tree.getroot()

    phases = []

    # 查找所有tlLogic节点中的phase元素
    for tl_logic in root.findall('tlLogic'):
        for phase_element in tl_logic.findall('phase'):
            # 提取duration,state,minDur,maxDur属性
            duration = int(phase_element.get('duration'))
            state = phase_element.get('state')
            minDur = int(phase_element.get('minDur'))
            maxDur = int(phase_element.get('maxDur'))

            # 创建Phase对象并添加到列表
            phase = Phase(duration, state, minDur, maxDur)
            phases.append(phase)

    return phases


class AtsSignalController:
    def __init__(self, ts_id: str):
        self.id = ts_id
        self.current_phase = 0
        self.last_change_time = time.time()
        self.phases = self._initialize_phases()

    def _initialize_phases(self) -> List[str]:
        # 在实际应用中，这里应该从配置文件或数据库中读取相位信息
        return [
            "GGGGGGGGggggggggrrrrrrrrrrrrrrrrGGGGGGGGggggggggrrrrrrrrrrrrrrrr",
            "yyyyyyyyggggggggrrrrrrrrrrrrrrrryyyyyyyyggggggggrrrrrrrrrrrrrrrr",
            "rrrrrrrrGGGGGGGGrrrrrrrrrrrrrrrrrrrrrrrrGGGGGGGGrrrrrrrrrrrrrrrr",
            "rrrrrrrryyyyyyyyrrrrrrrrrrrrrrrrrrrrrrrryyyyyyyyrrrrrrrrrrrrrrrr",
            "rrrrrrrrrrrrrrrrGGGGGGGGggggggggrrrrrrrrrrrrrrrrGGGGGGGGgggggggg",
            "rrrrrrrrrrrrrrrryyyyyyyyggggggggrrrrrrrrrrrrrrrryyyyyyyygggggggg",
            "rrrrrrrrrrrrrrrrrrrrrrrrGGGGGGGGrrrrrrrrrrrrrrrrrrrrrrrrGGGGGGGG",
            "rrrrrrrrrrrrrrrrrrrrrrrryyyyyyyyrrrrrrrrrrrrrrrrrrrrrrrryyyyyyyy"
        ]

    def _load_phases(self) -> List[Phase]:
        """Load phases from file and return"""
        file_path = "/Users/xnpeng/sumoptis/sumo-rl/zszx/net/zszx.tll.xml"
        return load_phases_from_file(file_path)

    def set_phase(self, phase_index: int):
        if 0 <= phase_index < len(self.phases):
            self.current_phase = phase_index
            self.last_change_time = time.time()
            self._send_signal_to_hardware(self.phases[phase_index])
        else:
            raise ValueError(f"Invalid phase index: {phase_index}")

    def _send_signal_to_hardware(self, phase_state: str):
        # 在实际应用中，这里应该实现与硬件的通信逻辑
        # 可能需要使用特定的协议（如NTCIP）来发送信号
        print(f"Sending signal to hardware for intersection {self.id}: {phase_state}")

    def get_current_phase(self) -> int:
        return self.current_phase

    def get_time_since_last_change(self) -> float:
        return time.time() - self.last_change_time

    def get_phase_duration(self, phase_index: int) -> int:
        # 在实际应用中，这里应该返回每个相位的实际持续时间
        # 这里我们简单地返回一个固定值
        return 30 if phase_index % 2 == 0 else 3  # 绿灯30秒，黄灯3秒

    def get_controlled_lanes(self) -> List[str]:
        # 在实际应用中，这里应该返回由这个信号控制的车道列表
        # 这里我们简单地返回一些示例车道
        return [f"{self.id}_lane_{i}" for i in range(1, 5)]

    def get_controlled_links(self) -> List[List[str]]:
        # 在实际应用中，这里应该返回由这个信号控制的链接列表
        # 这里我们简单地返回一些示例链接
        return [[f"{self.id}_in_lane_{i}", f"{self.id}_out_lane_{i}"] for i in range(1, 5)]

    def set_program(self, program: Dict):
        # 在实际应用中，这里应该设置新的信号程序
        # 这可能包括更新相位、时间等
        print(f"Setting new program for intersection {self.id}: {program}")

    def set_phase_duration(self, phase_index: int, duration: int):
        # 在实际应用中，这里应该更新特定相位的持续时间
        print(f"Setting duration for phase {phase_index} of intersection {self.id} to {duration} seconds")

    def close(self):
        print(f"Closing")
