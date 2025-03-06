"""Q-learning Agent class."""

from sumo_rl.agents.agent import Agent
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
from sumo_rl.environment.traffic_signal import TrafficSignal

class QLAgent(Agent):
  """Q-learning Agent class."""

  def __init__(self, id: str,
                     observation_fn: ObservationFunction,
                     controlled_entities: dict[str, TrafficSignal],
                     initial_states: dict,
                     state_space,
                     action_space,
                     alpha=0.5,
                     gamma=0.95,
                     exploration_strategy=EpsilonGreedy()):
    """Initialize Q-learning agent."""
    super().__init__(id, observation_fn)
    self.controlled_entities = controlled_entities
    self.state_space = state_space
    self.action_space = action_space

    self.previous_states = initial_states
    self.current_states = initial_states
    self.previous_actions = {}
    self.current_actions = {}

    self.q_table = {state: [0 for _ in range(action_space.n)] for state in self.previous_states.values()}
    self.alpha = alpha
    self.gamma = gamma
    self.exploration = exploration_strategy

  def reset(self, initial_states: dict):
    self.state = initial_states

  def observe(self):
    next_states = {}
    for ID in self.controlled_entities:
      controlled_entity = self.controlled_entities[ID]
      next_state = self.observation_fn(controlled_entity)
      next_states[ID] = next_state
      if next_state not in self.q_table:
        self.q_table[next_state] = [0 for _ in range(self.action_space.n)]
    self.previous_states = self.current_states
    self.current_states = next_states

  def act(self) -> dict[str, int]:
    """Choose action based on Q-table."""
    self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
    return self.action

  def learn(self, rewards: dict[str, float]):
    """Update Q-table with new experience."""

    for ID in self.controlled_entities.keys():
      previous_state = self.previous_states[ID]
      current_state = self.current_states[ID]
      previous_action = self.previous_actions[ID]
      reward = rewards[ID]
      self.q_table[previous_state][previous_action] = self.q_table[previous_state][previous_action] + self.alpha * (
        reward + self.gamma * max(self.q_table[current_state]) - self.q_table[previous_state][previous_action]
      )
