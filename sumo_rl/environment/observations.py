"""Observation functions for traffic signals."""

from abc import abstractmethod, ABC

import numpy
from gymnasium import spaces

from sumo_rl.environment.traffic_signal import TrafficSignal

class ObservationFunction(ABC):
    """Abstract base class for observation functions."""

    def __init__(self, name: str):
        """Initialize observation function."""
        self.name = name

    @abstractmethod
    def __call__(self, ts: TrafficSignal):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self, ts: TrafficSignal):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def hash(self, ts: TrafficSignal) -> str:
        """Subclasses must override this method."""
        pass

    def encode(self, state: numpy.ndarray, ts: TrafficSignal) -> tuple:
        """Encode the state of the traffic signal into a hashable object."""
        phase = int(numpy.where(state[: ts.num_green_phases] == 1)[0])
        min_green = state[ts.num_green_phases]
        density_queue = [self.discretize_density(d) for d in state[ts.num_green_phases + 1 :]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def discretize_density(self, density):
        return min(int(density * 10), 9)


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self):
        """Initialize default observation function."""
        super().__init__("default")

    def __call__(self, ts: TrafficSignal) -> tuple:
        """Return the default observation."""
        phase_id = [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 1]
        density = ts.get_lanes_density()
        queue = ts.get_lanes_queue()
        observation = numpy.array(phase_id + min_green + density + queue, dtype=numpy.float32)
        return self.encode(observation, ts)

    def observation_space(self, ts: TrafficSignal) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=numpy.zeros(ts.num_green_phases + 1 + 2 * len(ts.lanes), dtype=numpy.float32),
            high=numpy.ones(ts.num_green_phases + 1 + 2 * len(ts.lanes), dtype=numpy.float32),
        )

    def hash(self, ts: TrafficSignal):
        return "OS%s-%s-%sSO" % (self.name, ts.num_green_phases, len(ts.lanes))
