"""Reward functions for traffic signals."""

import sumo_rl.environment.traffic_signal
from sumo_rl.rewards import RewardFunction
from sumo_rl.environment.datastore import Datastore
import numpy

class DiffWaitingTimeRewardFunction(RewardFunction):
    """Diff waiting time reward function for traffic signals."""

    def __init__(self):
        """Initialize diff waiting time reward function."""
        super().__init__("diff-waiting-time")

    def __call__(self, datastore: Datastore, ts: sumo_rl.environment.traffic_signal.TrafficSignal) -> float:
        """Return the diff waiting time reward"""
        ts_wait = numpy.sum([datastore.lanes[lane_ID]['tawt'] for lane_ID in ts.lanes]) / 100.0
        reward = ts.last_ts_waiting_time - ts_wait
        ts.last_ts_waiting_time = ts_wait
        return reward
