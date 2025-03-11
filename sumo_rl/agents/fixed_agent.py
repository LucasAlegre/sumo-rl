"""Fixed Agent class."""

from sumo_rl.agents.agent import Agent
from sumo_rl.environment.traffic_signal import TrafficSignal

class FixedAgent(Agent):
  """Fixed Agent class."""

  def __init__(self, id: str,
                     controlled_entities: dict[str, TrafficSignal],
                     action_space):
    """Initialize Fixed agent."""
    super().__init__(id)
    self.controlled_entities = controlled_entities
    self.action_space = action_space
    self.previous_actions = {ID: 0 for ID in self.controlled_entities}
    self.current_actions = self.previous_actions
    self.steps_from_last_action = 0
    self.cycle_time_steps = 6

  def reset(self):
    self.previous_actions = {ID: 0 for ID in self.controlled_entities}
    self.current_actions = self.previous_actions
    self.steps_from_last_action = 0

  def hard_reset(self) -> None:
    self.reset()

  def observe(self):
    """Nothing is observed"""
    raise TypeError("FixedAgent doesn't support observing")

  def act(self) -> dict[str, int]:
    """Choose action cyclicly"""
    self.steps_from_last_action += 1
    if self.steps_from_last_action >= self.cycle_time_steps:
      actions = {}
      for ID in self.controlled_entities:
        actions[ID] = (self.previous_actions[ID] + 1) % self.action_space.n
      self.previous_actions = actions
      self.steps_from_last_action = 0
      return actions
    else:
      return self.previous_actions

  def learn(self):
    """Nothing is learned"""
    raise TypeError("FixedAgent doesn't support learning")

  def serialize(self, output_filepath: str) -> None:
    """Serialize Agent "memory" into an output file"""
    raise TypeError("FixedAgent doesn't support serialization/deserialization")

  def deserialize(self, input_filepath: str) -> None:
    """Deserialize Agent "memory" from an input file"""
    raise TypeError("FixedAgent doesn't support serialization/deserialization")
