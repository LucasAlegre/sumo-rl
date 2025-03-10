"""Observation functions for traffic signals."""

import numpy
import abc
from sumo_rl.environment.datastore import Datastore
import sumo_rl.environment.traffic_signal

class ObservationFunction(abc.ABC):
    """Abstract base class for observation functions."""

    def __init__(self, name: str):
        """Initialize observation function."""
        self.name = name

    @abc.abstractmethod
    def __call__(self, datastore: Datastore, ts: sumo_rl.environment.traffic_signal.TrafficSignal):
        """Subclasses must override this method."""
        pass

    @abc.abstractmethod
    def observation_space(self, ts: sumo_rl.environment.traffic_signal.TrafficSignal):
        """Subclasses must override this method."""
        pass

    @abc.abstractmethod
    def hash(self, ts: sumo_rl.environment.traffic_signal.TrafficSignal) -> str:
        """Subclasses must override this method."""
        pass

    def encode(self, state: numpy.ndarray, ts: sumo_rl.environment.traffic_signal.TrafficSignal) -> tuple:
        """Encode the state of the traffic signal into a hashable object."""
        phase = int(numpy.where(state[: ts.num_green_phases] == 1)[0])
        min_green = state[ts.num_green_phases]
        density_queue = [self.discretize_density(d) for d in state[ts.num_green_phases + 1 :]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def discretize_density(self, density):
        return min(int(density * 10), 9)
