"""Q-learning Agent class."""

import pickle
from sumo_rl.agents.agent import Agent
from sumo_rl.observations import ObservationFunction
from sumo_rl.rewards import RewardFunction
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
from sumo_rl.environment.traffic_signal import TrafficSignal

class QLAgent(Agent):
  """Q-learning Agent class."""

  def __init__(self, id: str,
                     observation_fn: ObservationFunction,
                     reward_fn: RewardFunction,
                     controlled_entities: dict[str, TrafficSignal],
                     state_space,
                     action_space,
                     alpha=0.5,
                     gamma=0.95,
                     exploration_strategy=EpsilonGreedy()):
    """Initialize Q-learning agent."""
    super().__init__(id)
    self.observation_fn: ObservationFunction = observation_fn
    self.reward_fn: RewardFunction = reward_fn
    self.controlled_entities = controlled_entities
    self.state_space = state_space
    self.action_space = action_space
    self.q_table = {}

    initial_states = self.compute_states()
    self.previous_states = initial_states
    self.current_states = initial_states
    self.previous_actions = {}
    self.current_actions = {}

    self.alpha = alpha
    self.gamma = gamma
    self.exploration = exploration_strategy

  def compute_states(self) -> dict:
    next_states = {}
    for ID in self.controlled_entities:
      controlled_entity = self.controlled_entities[ID]
      next_state = 0 # TODO self.observation_fn(controlled_entity)
      next_states[ID] = next_state
      if next_state not in self.q_table:
        self.q_table[next_state] = [0 for _ in range(self.action_space.n)]
    return next_states

  def compute_rewards(self) -> dict[str, float]:
    rewards = {}
    for ID in self.controlled_entities:
      reward = self.reward_fn(self.controlled_entities[ID])
      rewards[ID] = reward
    return rewards

  def reset(self, conn):
    for ID in self.controlled_entities:
      self.controlled_entities[ID].sumo = conn
    next_states = self.compute_states()
    self.previous_states = next_states
    self.current_states = next_states
    self.previous_actions = {}
    self.current_actions = {}

  def hard_reset(self, conn):
    self.q_table = {}
    self.reset(conn)

  def observe(self):
    next_states = self.compute_states()
    self.previous_states = self.current_states
    self.current_states = next_states

  def act(self) -> dict[str, int]:
    """Choose action based on Q-table."""
    actions = {}
    for ID in self.controlled_entities:
      state = self.current_states[ID]
      action = self.exploration.choose(self.q_table, state, self.action_space)
      actions[ID] = action
    self.previous_actions = actions
    return actions

  def learn(self):
    """Update Q-table with new experience."""
    rewards = self.compute_rewards()

    for ID in self.controlled_entities.keys():
      previous_state = self.previous_states[ID]
      current_state = self.current_states[ID]
      previous_action = self.previous_actions[ID]
      reward = rewards[ID]
      self.q_table[previous_state][previous_action] = self.q_table[previous_state][previous_action] + self.alpha * (
        reward + self.gamma * max(self.q_table[current_state]) - self.q_table[previous_state][previous_action]
      )

  def serialize(self, output_filepath: str) -> None:
    """Serialize Agent "memory" into an output file
    """
    with open(output_filepath, mode="wb") as file:
      pickle.dump(self.q_table, file)

  def deserialize(self, input_filepath: str) -> None:
    """Deserialize Agent "memory" from an input file
    """
    with open(input_filepath, mode="rb") as file:
      self.q_table = pickle.load(file)

  def __repr__(self) -> str:
    return "%s(%s)" % (self.__class__.__name__, list(self.controlled_entities.keys()))

  def can_be_serialized(self) -> bool:
    """True if serialization/deserialization is supported
    """
    return True

  def can_learn(self) -> bool:
    """True if learning is supported
    """
    return True
