#!/usr/bin/env python3
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
from sumo_rl.agents.ql_agent import QLAgent
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

class AgentFactory:
  def __init__(self, alpha, gamma, initial_epsilon, min_epsilon, decay) -> None:
    self.alpha: float = alpha
    self.gamma: float = gamma
    self.initial_epsilon: float = initial_epsilon
    self.min_epsilon: float = min_epsilon
    self.decay: float = decay

  def hash_traffic_signal_by_shape(self, observation_fn: ObservationFunction, traffic_signal: TrafficSignal) -> str:
    state_space_hash  = observation_fn.hash(traffic_signal)
    action_space_hash = str(traffic_signal.action_space)
    print(action_space_hash)
    return "SS%s-%sSS" % (state_space_hash, action_space_hash)
  
  def ql_agent_by_assignments(self, env: SumoEnvironment, assignments: dict[str, list[str]]) -> list[QLAgent]:
    agents = []
    initial_states = env.reset()
    traffic_signals = env.traffic_signals
    for agent_id, traffic_signal_ids in assignments.items():
      controlled_entities = {traffic_signal_id: traffic_signals[traffic_signal_id] for traffic_signal_id in traffic_signal_ids}
      initial_states = {traffic_signal_id: initial_states[traffic_signal_id] for traffic_signal_id in traffic_signal_ids}
      agents.append(self.ql_agent(agent_id, env.observation_fn, controlled_entities, initial_states))
    return agents
  
  def ql_agent_per_ts_shape(self, env: SumoEnvironment) -> list[QLAgent]:
    assignments = {}
    traffic_signals = env.traffic_signals
    for traffic_signal_id, traffic_signal in traffic_signals.items():
      hash = self.hash_traffic_signal_by_shape(env.observation_fn, traffic_signal)
      if hash not in assignments:
        assignments[hash] = []
      assignments[hash].append(traffic_signal_id)
    return self.ql_agent_by_assignments(env, assignments)
  
  def ql_agent_per_ts(self, env: SumoEnvironment) -> list[QLAgent]:
    assignments = {}
    traffic_signals = env.traffic_signals
    for traffic_signal_id in traffic_signals.keys():
      assignments[traffic_signal_id] = [traffic_signal_id]
    return self.ql_agent_by_assignments(env, assignments)

  def ql_agent(self, agent_id: str,
                     observation_fn: ObservationFunction,
                     controlled_entities: dict[str, TrafficSignal],
                     initial_states: dict) -> QLAgent:
    assert len(controlled_entities) > 0
    a_traffic_signal_id = list(controlled_entities)[0]
    action_space = controlled_entities[a_traffic_signal_id].action_space
    state_space = observation_fn.observation_space(controlled_entities[a_traffic_signal_id])
    return QLAgent(id=agent_id,
                   observation_fn=observation_fn,
                   controlled_entities=controlled_entities,
                   initial_states=initial_states,
                   state_space=state_space,
                   action_space=action_space,
                   alpha=self.alpha,
                   gamma=self.gamma,
                   exploration_strategy=EpsilonGreedy(initial_epsilon=self.initial_epsilon,
                                                      min_epsilon=self.min_epsilon,
                                                      decay=self.decay))
