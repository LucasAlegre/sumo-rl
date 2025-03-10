"""Reward functions for traffic signals."""

import sumo_rl.environment.traffic_signal
from sumo_rl.rewards import RewardFunction
from sumo_rl.environment.datastore import Datastore
import numpy

class QueueLengthRewardFunction(RewardFunction):
  """Queue length reward function for traffic signals."""

  def __init__(self):
    """Initialize queue length reward function."""
    super().__init__("queue-length")

  def __call__(self, datastore: Datastore, ts: sumo_rl.environment.traffic_signal.TrafficSignal) -> float:
    """Return the queue length reward"""
    return numpy.sum([datastore.lanes[lane_ID]['lshn'] for lane_ID in ts.lanes])
