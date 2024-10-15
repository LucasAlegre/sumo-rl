import os
import pygame
import sys
import time
from typing import Dict, List

sys.path.append('..')
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aits.RealWorldEnv import RealWorldEnv
from aits.RealWorldTrafficSignal import RealWorldTrafficSignal
from aits.RealWorldDataCollector import RealWorldDataCollector


class TrafficVisualization:
    def __init__(self, env: RealWorldEnv):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((1000, 800))
        pygame.display.set_caption("AITS - Traffic Signal Control")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((255, 255, 255))  # 白色背景
            self.draw_intersections()
            self.update_info_panel()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def draw_intersections(self):
        for i, ts_id in enumerate(self.env.intersection_ids):
            traffic_signal = self.env.traffic_signals[ts_id]
            data_collector = self.env.data_collectors[ts_id]
            x_offset = (i % 2) * 400 + 100
            y_offset = (i // 2) * 400 + 100
            self.draw_single_intersection(traffic_signal, data_collector, x_offset, y_offset)

    def draw_single_intersection(self, traffic_signal: RealWorldTrafficSignal,
                                 data_collector: RealWorldDataCollector, x_offset: int, y_offset: int):
        # 绘制道路
        pygame.draw.rect(self.screen, (200, 200, 200), (x_offset, y_offset, 200, 200))

        # 绘制信号灯
        current_phase = traffic_signal.signal_controller.get_current_phase()
        colors = self.get_phase_colors(current_phase)
        pygame.draw.circle(self.screen, colors[0], (x_offset + 90, y_offset + 90), 10)
        pygame.draw.circle(self.screen, colors[1], (x_offset + 110, y_offset + 110), 10)
        pygame.draw.circle(self.screen, colors[2], (x_offset + 90, y_offset + 110), 10)
        pygame.draw.circle(self.screen, colors[3], (x_offset + 110, y_offset + 90), 10)

        # 绘制车辆
        self.draw_vehicles(data_collector, x_offset, y_offset)

    def get_phase_colors(self, phase: int) -> List[tuple]:
        if phase == 0:  # "GGrrGGrr"
            return [(0, 255, 0), (255, 0, 0), (0, 255, 0), (255, 0, 0)]
        elif phase == 1:  # "yyrryyrr"
            return [(255, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 0)]
        elif phase == 2:  # "rrGGrrGG"
            return [(255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0)]
        else:  # "rryyrryy"
            return [(255, 0, 0), (255, 255, 0), (255, 0, 0), (255, 255, 0)]

    def draw_vehicles(self, data_collector: RealWorldDataCollector, x_offset: int, y_offset: int):
        lane_positions = {
            f"{data_collector.intersection_id}_lane_1": (x_offset, y_offset + 80, 80, 20),
            f"{data_collector.intersection_id}_lane_2": (x_offset + 100, y_offset + 100, 80, 20),
            f"{data_collector.intersection_id}_lane_3": (x_offset + 120, y_offset, 20, 80),
            f"{data_collector.intersection_id}_lane_4": (x_offset + 100, y_offset + 120, 20, 80)
        }

        for lane, (x, y, w, h) in lane_positions.items():
            vehicle_count = data_collector.get_lane_vehicle_count(lane)
            for i in range(vehicle_count):
                if w > h:  # 水平车道
                    veh_x = x + (i * 20) % w
                    veh_y = y
                else:  # 垂直车道
                    veh_x = x
                    veh_y = y + (i * 20) % h
                pygame.draw.rect(self.screen, (0, 0, 255), (veh_x, veh_y, 10, 10))

    def update_info_panel(self):
        y = 10
        for ts_id in self.env.intersection_ids:
            data_collector = self.env.data_collectors[ts_id]
            text = self.font.render(f"Intersection {ts_id}:", True, (0, 0, 0))
            self.screen.blit(text, (720, y))
            y += 30

            total_vehicles = sum(data_collector.get_lane_vehicle_count(lane) for lane in data_collector.lanes)
            text = self.font.render(f"Total Vehicle Number: {total_vehicles}", True, (0, 0, 0))
            self.screen.blit(text, (720, y))
            y += 30

            avg_speed = data_collector.get_average_speed()
            text = self.font.render(f"Average Speed: {avg_speed:.2f} m/s", True, (0, 0, 0))
            self.screen.blit(text, (720, y))
            y += 30

            total_queued = data_collector.get_total_queued()
            text = self.font.render(f"Queued Vehicle: {total_queued}", True, (0, 0, 0))
            self.screen.blit(text, (720, y))
            y += 50


def main(action_space_type="auto"):
    env = RealWorldEnv(
        intersection_ids=["intersection_1", "intersection_2"],
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=30,
        num_seconds=3600,
        reward_fn="queue",
        action_space_type=action_space_type
    )

    env.reset()

    vis = TrafficVisualization(env)

    def update_env():
        action = env.action_space.sample()
        env.step(action)

    # 创建一个 Pygame 事件来定期更新环境
    UPDATE_ENV_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(UPDATE_ENV_EVENT, 1000)  # 每秒更新一次环境

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == UPDATE_ENV_EVENT:
                update_env()

        vis.run()

    env.close()


if __name__ == "__main__":
    print("test main-ui")
    main()
