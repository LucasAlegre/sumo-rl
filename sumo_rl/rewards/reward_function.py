"""Reward functions for traffic signals."""

import abc
import sumo_rl.environment.traffic_signal
from sumo_rl.environment.datastore import Datastore

class RewardFunction(abc.ABC):
    """Abstract base class for reward functions."""

    def __init__(self, name: str):
        """Initialize reward function."""
        self.name = name

    @abc.abstractmethod
    def __call__(self, datastore: Datastore, ts: sumo_rl.environment.traffic_signal.TrafficSignal):
        """Subclasses must override this method."""
        pass

