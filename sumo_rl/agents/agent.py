"""Abstract Agent class."""

import abc
from sumo_rl.environment.observations import ObservationFunction

class Agent(abc.ABC):
    """Abstract Agent class.

    An Agent should be able to comand multiple entities with the same state-space and action-space.
    """

    def __init__(self, id: str, observation_fn: ObservationFunction) -> None:
      """Initializes the Agent to use the ObservationFunction described"""
      self.id: str = id
      self.observation_fn: ObservationFunction = observation_fn

    @abc.abstractmethod
    def reset(self, initial_states: dict) -> None:
      """Resets the agent's view of simulation
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
    def learn(self, rewards: dict[str, float]) -> None:
      """Learn from errors
      
      It should have cached its previous action and states and use them to understand what it has done.
      Expectes a list of rewards for its controlled entities.
      """
      pass
