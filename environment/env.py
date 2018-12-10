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
        self.timeOnPhase = 0
        self.edges = self._compute_edges()
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
        self.timeOnPhase += time_keep
        traci.trafficlight.setPhaseDuration(self.id, time_keep)

    def change(self):
        self.timeOnPhase = 0
        traci.trafficlight.setPhaseDuration(self.id, 0)

    def _compute_edges(self):
        """
        :return: Dict green phase to edge id
        """

        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))
        return {0: lanes[:2], 3: lanes[2:]}


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
        self.radix_factors = [s.n for s in self.observation_space.spaces]

    def reset(self):
        sumo_cmd = [self._sumo_binary, '-c', self._conf]
        traci.start(sumo_cmd)
        self.ts_ids = traci.trafficlight.getIDList()
        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(ts)

        # Load vehicles
        for _ in range(300):
            traci.simulationStep

        return

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

        observation = dict()
        reward = dict()
        for ts in self.ts_ids:
            observation[ts] = 1
        done = self.sim_step > self.sim_max_time

        return done

    def _radix_encode(self):


    def close(self):
        traci.close()
