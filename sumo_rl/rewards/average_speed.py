"""Reward functions for traffic signals."""

import sumo_rl.environment.traffic_signal
from sumo_rl.rewards import RewardFunction
from sumo_rl.environment.datastore import Datastore
import numpy

class AverageSpeedRewardFunction(RewardFunction):
    """Average speed reward function for traffic signals."""

    def __init__(self):
        """Initialize average speed reward function."""
        super().__init__("average-speed")

    def __call__(self, datastore: Datastore, ts: sumo_rl.environment.traffic_signal.TrafficSignal) -> float:
        """Return the average speed reward"""
        return numpy.mean([datastore.lanes[lane_ID]['lsms']/datastore.lanes[lane_ID]['ms'] for lane_ID in ts.lanes])
