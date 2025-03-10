"""Reward functions for traffic signals."""

import sumo_rl.environment.traffic_signal
from sumo_rl.rewards import RewardFunction
from sumo_rl.environment.datastore import Datastore
import numpy

class PressureRewardFunction(RewardFunction):
    """Pressure reward function for traffic signals."""

    def __init__(self):
        """Initialize pressure reward function."""
        super().__init__("pressure")

    def __call__(self, datastore: Datastore, ts: sumo_rl.environment.traffic_signal.TrafficSignal) -> float:
        """Return the pressure reward"""
        return numpy.sum([datastore.lanes[lane_ID]['lsvn'] for lane_ID in ts.out_lanes]) - numpy.sum([datastore.lanes[lane_ID]['lsvn'] for lane_ID in ts.lanes])
