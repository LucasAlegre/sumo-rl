import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class Info:
    def __init__(self):
        self.label = "csvfile"
        self.out_csv_name = "my_results"
        self.num_episodes = 10
        self.sim_step = 5
        self.episode = 0
        self.metrics = []
        self.ts_ids = ["ts0", "ts1", "ts2", "ts3", "ts4", "ts5", "ts6", "ts7"]


    def show(self):
        self._compute_info()
        print(self.metrics)
        self.save_csv(self.out_csv_name, self.episode)

    def _compute_info(self):

        info = {"step": self.sim_step}
        info.update(self._get_system_info())
        info.update(self._get_per_agent_info())
        self.metrics.append(info)
        return info

    def _get_system_info(self):
        vehicles = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
        speeds = [0.0, 0.1, 2.0, 3.0, 4.0, 5.0]
        waiting_times = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    def _get_per_agent_info(self):
        stopped = [0, 1, 2, 3, 4, 5, 6, 7]
        accumulated_waiting_time = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
        average_speed = [2.0, 5.0, 7.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)


if __name__ == "__main__":
    info = Info()
    info.show()
