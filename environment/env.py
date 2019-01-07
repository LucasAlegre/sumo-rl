import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from gym import Env
import traci.constants as tc
from gym import spaces
import numpy as np


from .traffic_signal import TrafficSignal


class SumoEnvironment(Env):

    KEEP = 0
    CHANGE = 1

    def __init__(self, conf_file,
                 use_gui=False,
                 num_seconds=20000,
                 max_depart_delay=100000,
                 time_to_load_vehicles=0,
                 delta_time=5,
                 min_green=10,
                 max_green=50,
                 custom_phases=None):

        self._conf = conf_file
        if use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.ts_ids = list()
        self.traffic_signals = dict()
        self.custom_phases = custom_phases
        self.last_measure = dict()    # used to reward function remember last measure
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles  # number of seconds of simulation ran in reset()
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.min_green = min_green
        self.max_green = max_green
        self.total_load_vehicles = 0
        self.total_departed_vehicles = 0

        self.observation_space = spaces.Tuple((
            spaces.Discrete(2),   # Phase NS or EW
            spaces.Discrete(self.max_green//self.delta_time),  # Elapsed time of phase
            spaces.Discrete(10),  # NS density
            spaces.Discrete(10),  # EW density
            spaces.Discrete(10),  # NS stopped-density
            spaces.Discrete(10))  # EW stopped-density
        )
        self.action_space = spaces.Discrete(2)  # Keep or change

        self.radix_factors = [s.n for s in self.observation_space.spaces]

    def reset(self):
        sumo_cmd = [self._sumo_binary, '-c', self._conf, '--max-depart-delay', str(self.max_depart_delay), '--waiting-time-memory', '10000', '--random']
        traci.start(sumo_cmd)

        self.ts_ids = traci.trafficlight.getIDList()
        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(ts, self.delta_time, self.min_green, self.max_green, self.custom_phases)
            self.last_measure[ts] = 0.0
        TrafficSignal.vehicles = {}

        self.total_load_vehicles = 0
        self.total_departed_vehicles = 0

        # Load vehicles
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()

        return self._compute_observations()

    @property
    def sim_step(self):
        return traci.simulation.getCurrentTime()/1000  # milliseconds to seconds

    def step(self, actions):
        # act
        self.apply_actions(actions)

        # run simulation for delta time
        for _ in range(self.delta_time):
            self._sumo_step()

        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        done = self.sim_step > self.sim_max_time

        return observation, reward, done, self._compute_step_info()

    def apply_actions(self, actions):
        for ts, action in actions.items():
            if action == self.KEEP:
                self.traffic_signals[ts].keep()
            elif action == self.CHANGE:
                self.traffic_signals[ts].change()
            else:
                exit('Invalid action!')

    def _compute_observations(self):
        observations = {}
        for ts in self.ts_ids:
            phase_id = self.traffic_signals[ts].phase // 2  # 0 -> 0 and 2 -> 1
            elapsed = self._discretize_elapsed_time(self.traffic_signals[ts].time_on_phase)

            ns_density, ew_density = self.traffic_signals[ts].get_density()
            ns_density, ew_density = self._discretize_density(ns_density), self._discretize_density(ew_density)

            ns_stop_density, ew_stop_density = self.traffic_signals[ts].get_stopped_density()
            ns_stop_density, ew_stop_density = self._discretize_density(ns_stop_density), self._discretize_density(ew_stop_density)

            observations[ts] = self.radix_encode([phase_id, elapsed, ns_density, ew_density, ns_stop_density, ew_stop_density])
        return observations

    def _compute_rewards(self):
        return self._waiting_time_reward()
        #return self._waiting_time_reward2()
        #return self._queue_average_reward()

    def _queue_average_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            ns_stopped, ew_stopped = self.traffic_signals[ts].get_stopped_vehicles_num()
            new_average = (ns_stopped + ew_stopped) / 2
            rewards[ts] = self.last_measure[ts] - new_average
            self.last_measure[ts] = new_average
        return rewards

    def _waiting_time_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            ns_wait, ew_wait = self.traffic_signals[ts].get_waiting_time()
            ts_wait = ns_wait + ew_wait
            rewards[ts] = self.last_measure[ts] - ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _waiting_time_reward2(self):
        rewards = {}
        for ts in self.ts_ids:
            ns_wait, ew_wait = self.traffic_signals[ts].get_waiting_time()
            ts_wait = ns_wait + ew_wait
            if ts_wait == 0:
                rewards[ts] = 1.0
            else:
                rewards[ts] = 1.0/ts_wait
        return rewards

    def _discretize_density(self, density):
        if density < 0.1:
            return 0
        elif density < 0.2:
            return 1
        elif density < 0.3:
            return 2
        elif density < 0.4:
            return 3
        elif density < 0.5:
            return 4
        elif density < 0.6:
            return 5
        elif density < 0.7:
            return 6
        elif density < 0.8:
            return 7
        elif density < 0.9:
            return 8
        else:
            return 9

    def _discretize_elapsed_time(self, elapsed):
        for i in range(self.max_green//self.delta_time):
            if elapsed <= self.delta_time + i*self.delta_time:
                return i

    def radix_encode(self, values):
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

    def radix_decode(self, value):
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res

    def _sumo_step(self):
        traci.simulationStep()
        # self.total_load_vehicles += traci.simulation.getLoadedNumber()
        # self.total_departed_vehicles += traci.simulation.getDepartedNumber()

    @property
    def buffer_size(self):
        return self.total_load_vehicles - self.total_departed_vehicles

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'total_stopped': sum([sum(self.traffic_signals[ts].get_stopped_vehicles_num()) for ts in self.ts_ids]),
            'total_wait_time': sum([sum(self.traffic_signals[ts].get_waiting_time()) for ts in self.ts_ids]),
            # 'buffer_size': self.buffer_size
        }

    def close(self):
        traci.close()

