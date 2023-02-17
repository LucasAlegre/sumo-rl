from .traffic_signal import TrafficSignal
from abc import abstractmethod
from typing import Callable
from gymnasium import spaces
import numpy as np

class ObservationFunction:
    """
    Abstract base class for observation functions.
    """
    def __init__(self, ts: TrafficSignal):
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """
        Subclasses must override this method.
        """
        pass

    @abstractmethod
    def observation_space(self):
        """
        Subclasses must override this method.
        """
        pass


class DefaultObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        # super().__init__(ts)
        self.ts = ts

    def __call__(self):
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self):
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases+1+2*len(self.ts.lanes), dtype=np.float32), 
            high=np.ones(self.ts.num_green_phases+1+2*len(self.ts.lanes), dtype=np.float32)
        )

