"""Abstract Agent class."""

import abc
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.rewards import RewardFunction

class Agent(abc.ABC):
    """Abstract Agent class.

    An Agent should be able to comand multiple entities with the same state-space and action-space.
    """

    def __init__(self, id: str) -> None:
      """Initializes the Agent"""
      self.id: str = id

    @abc.abstractmethod
    def reset(self, conn) -> None:
      """Resets the agent's view of simulation
      """
      pass

    @abc.abstractmethod
    def hard_reset(self, conn) -> None:
      """Resets the agent's view of simulation and its memory
      """
      pass

    @abc.abstractmethod
    def observe(self) -> None:
      """Feedback of simulation

      Uses controlled entities to compute observations
      """
      pass

    @abc.abstractmethod
    def act(self) -> dict[str, int]:
      """Choose an action based on the current state of things.

      Returns the list of actions of controlled entities.
      """
      pass

    @abc.abstractmethod
    def learn(self) -> None:
      """Learn from errors
      
      It should have cached its previous action and states and use them to understand what it has done.
      Uses controlled entities to obtain rewards
      """
      pass

    @abc.abstractmethod
    def serialize(self, output_filepath: str) -> None:
      """Serialize Agent "memory" into an output file
      
      Its thought to be used with deserialize()
      Subclasses which don't support serialization/deserialization should throw a TypeError
      """
      pass

    @abc.abstractmethod
    def deserialize(self, input_filepath: str) -> None:
      """Deserialize Agent "memory" from an input file
      
      Its thought to be used with serialize()
      Subclasses which don't support serialization/deserialization should throw a TypeError
      """
      pass
