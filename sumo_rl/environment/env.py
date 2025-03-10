"""SUMO Environment for Traffic Signal Control."""

import os
import sys
import numpy
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from sumo_rl.environment.datastore import Datastore



if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci

import sumo_rl.observations
import sumo_rl.rewards
from .traffic_signal import TrafficSignal


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

class SumoEnvironment(gym.Env):
    """SUMO Environment for Traffic Signal Control.

    Class that implements a gym.Env interface for traffic signal control using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The duration in seconds of the simulation. Default: 20000
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        enforce_max_green (bool): If true, it enforces the max green time and selects the next green phase when the max green time is reached. Default: False
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict/List): String with the name of the reward function used by the agents, a reward function, dictionary with reward functions assigned to individual traffic lights by their keys, or a List of reward functions.
        reward_weights (List[float]/np.ndarray): Weights for linearly combining the reward functions, in case reward_fn is a list. If it is None, the reward returned will be a np.ndarray. Default: None
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        enforce_max_green: bool = False,
        single_agent: bool = False,
        observation_fn: sumo_rl.observations.ObservationFunction = sumo_rl.observations.DefaultObservationFunction(),
        reward_fn: sumo_rl.rewards.RewardFunction = sumo_rl.rewards.AverageSpeedRewardFunction(),
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."
        assert max_green > min_green, "Max green time must be greater than min green time."

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.enforce_max_green = enforce_max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net], label=self.label)
            conn = traci.getConnection(self.label)

        self.ts_ids = list(conn.trafficlight.getIDList())
        self.observation_fn = observation_fn
        self.reward_fn = reward_fn

        self._build_traffic_signals(conn)

        self.sumo = conn
        # conn.close()

        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.out_csv_name = out_csv_name

        # USIAMOLE
        assert len(self.ts_ids) == len(self.traffic_signals)

        self.datastore = Datastore()
        self.datastore.lanes = {lane_ID:{'ms': self.sumo.lane.getMaxSpeed(lane_ID)} for ts_id in self.traffic_signals for lane_ID in self.sumo.trafficlight.getControlledLanes(ts_id)}
        self.observations = {ts_id:[] for ts_id in self.traffic_signals}
        self.rewards = {ts: None for ts in self.ts_ids}
        self.metrics = self.empty_metrics()

    def _build_traffic_signals(self, conn):
        self.traffic_signals = {
            ts: TrafficSignal(
                self,
                ts,
                self.delta_time,
                self.yellow_time,
                self.min_green,
                self.max_green,
                self.enforce_max_green,
                self.begin_time,
                conn,
            )
            for ts in self.ts_ids
        }

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            if "DEFAULT_VIEW" not in dir(traci.gui):  # traci.gui.DEFAULT_VIEW is not defined in libsumo
                traci.gui.DEFAULT_VIEW = "View #0"
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)

        self.close()
        self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = self.empty_metrics()

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        for ts_id in self.traffic_signals:
          self.traffic_signals[ts_id].reset(self.begin_time)

        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0

        return self.sumo

    def gather_data_from_sumo(self):
      vehicles = set({})
      for lane_ID in self.datastore.lanes.keys():
        self.datastore.lanes[lane_ID]['lsvn'] = self.sumo.lane.getLastStepVehicleNumber(lane_ID)
        self.datastore.lanes[lane_ID]['lsvl'] = self.sumo.lane.getLastStepLength(lane_ID)
        self.datastore.lanes[lane_ID]['lshn'] = self.sumo.lane.getLastStepHaltingNumber(lane_ID)
        self.datastore.lanes[lane_ID]['lsms'] = self.sumo.lane.getLastStepMeanSpeed(lane_ID)
        self.datastore.lanes[lane_ID]['lso'] = self.sumo.lane.getLastStepOccupancy(lane_ID)
        self.datastore.lanes[lane_ID]['vehs'] = set(self.sumo.lane.getLastStepVehicleIDs(lane_ID))
        vehicles |= self.datastore.lanes[lane_ID]['vehs']
      self.datastore.vehicles = {
        vehicle_ID: {
          's': self.sumo.vehicle.getSpeed(vehicle_ID),
          'wt': self.sumo.vehicle.getWaitingTime(vehicle_ID),
          'awt': self.sumo.vehicle.getAccumulatedWaitingTime(vehicle_ID),
        }
        for vehicle_ID in vehicles
      }
      for lane_ID in self.datastore.lanes.keys():
        awts = [self.datastore.vehicles[vehicle_ID]['awt'] for vehicle_ID in self.datastore.lanes[lane_ID]['vehs']]
        self.datastore.lanes[lane_ID]['tawt'] = numpy.sum(awts)
        self.datastore.lanes[lane_ID]['mawt'] = numpy.mean(awts)

    def compute_observations(self):
      for ts_ID, ts in self.traffic_signals.items():
        self.observations[ts_ID] = self.observation_fn(self.datastore, ts)

    def compute_rewards(self):
      for ts_ID, ts in self.traffic_signals.items():
        self.rewards[ts_ID] = self.reward_fn(self.datastore, ts)

    def empty_metrics(self) -> dict:
      return {
        "step": [],
        "total_running": [],
        "total_backlogged": [],
        "total_stopped": [],
        "total_arrived": [],
        "total_departed": [],
        "total_teleported": [],
        "total_waiting_time": [],
        "mean_waiting_time": [],
        "mean_speed": [],
      }

    def compute_metrics(self):
      self.metrics["step"].append(self.sim_step)
      self.metrics["total_running"].append(len(self.datastore.vehicles))
      self.metrics["total_backlogged"].append(len(self.sumo.simulation.getPendingVehicles()))
      self.metrics["total_stopped"].append(numpy.sum([self.datastore.vehicles[vehicle_ID]['s'] < 0.1 for vehicle_ID in self.datastore.vehicles]))
      self.metrics["total_arrived"].append(self.num_arrived_vehicles)
      self.metrics["total_departed"].append(self.num_departed_vehicles)
      self.metrics["total_teleported"].append(self.num_teleported_vehicles)
      self.metrics["total_waiting_time"].append(numpy.sum([self.datastore.vehicles[vehicle_ID]['wt'] for vehicle_ID in self.datastore.vehicles]))
      self.metrics["mean_waiting_time"].append(numpy.mean([self.datastore.vehicles[vehicle_ID]['wt'] for vehicle_ID in self.datastore.vehicles]))
      self.metrics["mean_speed"].append(numpy.mean([self.datastore.vehicles[vehicle_ID]['s'] for vehicle_ID in self.datastore.vehicles]))

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()

    def done(self) -> bool:
      return self.sim_step >= self.sim_max_time

    def step(self, action: dict):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        self._apply_actions(action)
        for _ in range(self.delta_time):
          self._sumo_step()
          for ts in self.ts_ids:
              self.traffic_signals[ts].update()

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        """Set the next green phase for the traffic signals.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {ts_id : greenPhase}
        """
        for ts, action in actions.items():
            if self.traffic_signals[ts].time_to_act:
                self.traffic_signals[ts].set_next_phase(action)

    def _sumo_step(self):
        self.sumo.simulationStep()
        self.num_arrived_vehicles += self.sumo.simulation.getArrivedNumber()
        self.num_departed_vehicles += self.sumo.simulation.getDepartedNumber()
        self.num_teleported_vehicles += self.sumo.simulation.getEndingTeleportNumber()

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        num_backlogged_vehicles = len(self.sumo.simulation.getPendingVehicles())
        return {
            "system_total_running": len(vehicles),
            "system_total_backlogged": num_backlogged_vehicles,
            "system_total_stopped": sum(
                int(speed < 0.1) for speed in speeds
            ),  # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_arrived": self.num_arrived_vehicles,
            "system_total_departed": self.num_departed_vehicles,
            "system_total_teleported": self.num_teleported_vehicles,
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
      }

    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids
        ]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        if self.sumo is None:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self.close()

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
