import os
import sys
import traci
import sumolib
from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
import traci
import traci.constants as tc
from scipy.misc import imread
from gym import spaces
from string import Template
import os, sys
import numpy as np
import math
import time
import re


class TrafficSignal:

    def __init__(self, id):
        self.id = id
        self.time_on_phase = 0
        self.edges = self._compute_edges()
        self.ns_stopped = [0, 0]
        self.ew_stopped = [0, 0]
        phases = [
            traci.trafficlight.Phase(35000, 35000, 35000, "GGGrrr"),   # norte-sul - 0
            traci.trafficlight.Phase(2000, 2000, 2000, "yyyrrr"),
            traci.trafficlight.Phase(1, 1, 1, "rrrrrr"),
            traci.trafficlight.Phase(35000, 35000, 35000, "rrrGGG"),   # leste-oeste - 3
            traci.trafficlight.Phase(2000, 2000, 2000, "rrryyy"),
            traci.trafficlight.Phase(1, 1, 1, "rrrrrr")
        ]
        logic = traci.trafficlight.Logic("new-program", 0, 0, 0, phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def keep(self, time_keep):
        self.time_on_phase += time_keep
        traci.trafficlight.setPhaseDuration(self.id, time_keep)

    def change(self):
        self.time_on_phase = 0
        traci.trafficlight.setPhaseDuration(self.id, 0)

    def _compute_edges(self):
        """
        :return: Dict green phase to edge id
        """
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))
        return {0: lanes[:2], 3: lanes[2:]}

    def compute_stopped_vehicles_edge(self):
        self.ns_stopped[1], self.ew_stopped[1] = self.ns_stopped[0], self.ew_stopped[0]
        self.ns_stopped[0] = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[0]])
        self.ew_stopped[0] = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[3]])
        return self.ns_stopped[0], self.ew_stopped[0]


class SumoEnvironment(Env):

    KEEP = 0
    CHANGE = 1

    def __init__(self, conf_file, use_gui=False, num_seconds=20000):
        self._conf = conf_file
        if use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.ts_ids = list()
        self.traffic_signals = dict()
        self.sim_max_time = num_seconds
        self.delta_time = 5  # seconds on sumo at each step

        self.observation_space = spaces.Tuple((
            spaces.Discrete(2),  # Phase NS or EW
            spaces.Discrete(5),  # Duration of phase
            spaces.Discrete(4),  # NS stopped cars
            spaces.Discrete(4))  # EW stopped cars
        )
        self.action_space = spaces.Discrete(2)  # Keep or change

        self.actual_observation = None

        self.radix_factors = [s.n for s in self.observation_space.spaces]

    def reset(self):
        sumo_cmd = [self._sumo_binary, '-c', self._conf]
        traci.start(sumo_cmd)
        self.ts_ids = traci.trafficlight.getIDList()
        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(ts)

        # Load vehicles
        for _ in range(300):
            traci.simulationStep()

        self.actual_observation = self._compute_observations()
        return self.actual_observation

    @property
    def sim_step(self):
        #print(traci.simulation.getDeltaT())
        return traci.simulation.getCurrentTime()/1000

    def step(self, actions):

        for ts, action in actions:
            if action == self.KEEP:
                self.traffic_signals[ts].keep(self.delta_time)
            elif action == self.CHANGE:
                self.traffic_signals[ts].change()

        for _ in range(self.delta_time):
            traci.simulationStep()

        observation = self._compute_observations()
        reward = self._compute_rewards()
        done = self.sim_step > self.sim_max_time

        return observation, reward, done, None

    def _radix_encode(self, phase_id, duration, ns_stopped, ew_stopped):
        values = [phase_id, duration, ns_stopped, ew_stopped]
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]

        return res

    def _compute_observations(self):
        observations = {}
        for ts in self.ts_ids:
            phase_id = self.traffic_signals[ts].phase
            duration = self.traffic_signals[ts].time_on_phase / 5
            ns_stopped, ew_stopped = self.traffic_signals[ts].compute_stopped_vehicles_edge()

            ns_stopped = ns_stopped / 40
            ns_class = math.ceil(ns_stopped / 25)
            if ns_stopped >= 75:
                ns_class = 3
            ew_stopped = ew_stopped / 40
            ew_class = math.ceil(ew_stopped / 25)
            if ew_stopped >= 75:
                ew_class = 3

            observations[ts] = self._radix_encode(phase_id, duration, int(ns_class), int(ew_class))
        return observations

    def _compute_rewards(self):
        rewards = {}
        for ts in self.ts_ids:
            old_average = ((self.traffic_signals[ts].nw_stopped[1] + self.traffic_signals[ts].ew_stopped[1]) / 2)
            new_average = ((self.traffic_signals[ts].nw_stopped[0] + self.traffic_signals[ts].ew_stopped[0]) / 2)
            rewards[ts] = old_average - new_average
        return rewards

    def close(self):
        traci.close()
