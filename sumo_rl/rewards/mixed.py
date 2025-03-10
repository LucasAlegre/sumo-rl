"""Reward functions for traffic signals."""

import numpy
import sumo_rl.environment.traffic_signal
from sumo_rl.rewards import RewardFunction
from sumo_rl.environment.datastore import Datastore

class MixedRewardFunction(RewardFunction):
    """Mixed reward function for traffic signals."""

    def __init__(self, reward_fns: list[RewardFunction], weights: list[float] = None):
        """Initialize pressure reward function."""
        assert len(reward_fns) > 0
        super().__init__("--".join([reward_fn.name for reward_fn in reward_fns]))
        self.reward_fns: list[RewardFunction] = reward_fns

        if weights is None:
          length = len(self.reward_fns)
          self.weights = [1/length for _ in range(length)]
        assert(len(self.reward_fns) == len(self.weights))
        assert(sum(self.weights) == 1)

    def __call__(self, datastore: Datastore, ts: sumo_rl.environment.traffic_signal.TrafficSignal) -> float:
        """Return the pressure reward"""
        rewards = [reward_fn(datastore, ts) for reward_fn in self.reward_fns]
        if len(rewards) > 1:
          return numpy.dot(rewards, self.weights)
        return rewards[0]
