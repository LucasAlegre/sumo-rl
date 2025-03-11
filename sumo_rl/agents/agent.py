"""Abstract Agent class."""

import abc
import typing

class Agent(abc.ABC):
    """Abstract Agent class.

    An Agent should be able to comand multiple entities with the same state-space and action-space.
    """

    def __init__(self, id: str) -> None:
      """Initializes the Agent"""
      self.id: str = id

    @abc.abstractmethod
    def reset(self) -> None:
      """Resets the agent's view of simulation
      """
      pass

    @abc.abstractmethod
    def hard_reset(self) -> None:
      """Resets the agent's view of simulation and its memory
      """
      pass

    @abc.abstractmethod
    def observe(self, observations: dict[str, typing.Any]) -> None:
      """Feedback of simulation

      Provides observations of controlled entities.
      Subclasses which don't support learning should throw a TypeError
      """
      pass

    @abc.abstractmethod
    def act(self) -> dict[str, int]:
      """Choose an action based on the current state of things.

      Returns the list of actions of controlled entities.
      """
      pass

    @abc.abstractmethod
    def learn(self, rewards: dict[str, typing.Any]) -> None:
      """Learn from errors
      
      It should have cached its previous action and states and use them to understand what it has done.
      Provides rewards of controlled entities.
      Subclasses which don't support learning should throw a TypeError
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

    def can_be_serialized(self) -> bool:
      """True if serialization/deserialization is supported
      """
      return False

    def can_learn(self) -> bool:
      """True if learning is supported
      """
      return False

    def can_observe(self) -> bool:
      """True if observing is supported
      """
      return False

    def __repr__(self) -> str:
      """String representation at runtime of agent (shouldn't contain implementation details or "memory"
      """
      return self.__class__.__name__
