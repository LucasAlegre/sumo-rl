"""Observation functions for traffic signals."""

from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from sumo_rl.environment.traffic_signal import TrafficSignal

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self):
        """Initialize observation function."""

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

    @abstractmethod
    def name(self) -> str:
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self):
        """Initialize default observation function."""
        super().__init__()

    def __call__(self, ts: TrafficSignal) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 1]
        density = ts.get_lanes_density()
        queue = ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self, ts: TrafficSignal) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(ts.num_green_phases + 1 + 2 * len(ts.lanes), dtype=np.float32),
            high=np.ones(ts.num_green_phases + 1 + 2 * len(ts.lanes), dtype=np.float32),
        )

    def hash(self, ts: TrafficSignal):
        return "OS%s-%s-%sSO" % (self.name(), ts.num_green_phases, len(ts.lanes))

    def name(self):
        return "default"
