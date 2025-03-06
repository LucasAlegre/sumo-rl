"""Reward functions for traffic signals."""

from abc import abstractmethod, ABC

from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy

class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    def __init__(self, name: str):
        """Initialize reward function."""
        self.name = name

    @abstractmethod
    def __call__(self, ts: TrafficSignal):
        """Subclasses must override this method."""
        pass

class DiffWaitingTimeRewardFunction(RewardFunction):
    """Diff waiting time reward function for traffic signals."""

    def __init__(self):
        """Initialize diff waiting time reward function."""
        super().__init__("diff-waiting-time")

    def __call__(self, ts: TrafficSignal) -> float:
        """Return the diff waiting time reward"""
        ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = ts.last_ts_waiting_time - ts_wait
        ts.last_ts_waiting_time = ts_wait
        return reward

class AverageSpeedRewardFunction(RewardFunction):
    """Average speed reward function for traffic signals."""

    def __init__(self):
        """Initialize average speed reward function."""
        super().__init__("average-speed")

    def __call__(self, ts: TrafficSignal) -> float:
        """Return the average speed reward"""
        return ts.get_average_speed()

class QueueLengthFunction(RewardFunction):
    """Queue length reward function for traffic signals."""

    def __init__(self):
        """Initialize queue length reward function."""
        super().__init__("queue-length")

    def __call__(self, ts: TrafficSignal) -> float:
        """Return the queue length reward"""
        return -ts.get_total_queued()

class PressureRewardFunction(RewardFunction):
    """Pressure reward function for traffic signals."""

    def __init__(self):
        """Initialize pressure reward function."""
        super().__init__("pressure")

    def __call__(self, ts: TrafficSignal) -> float:
        """Return the pressure reward"""
        return ts.get_pressure()

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

    def __call__(self, ts: TrafficSignal) -> float:
        """Return the pressure reward"""
        rewards = [reward_fn(ts) for reward_fn in self.reward_fns]
        if len(rewards) > 1:
          return numpy.dot(rewards, self.weights)
        return rewards[0]
