#!/usr/bin/env python3
import abc
import os
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.rewards import RewardFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
from sumo_rl.agents.agent import Agent
from sumo_rl.agents.ql_agent import QLAgent
from sumo_rl.agents.fixed_agent import FixedAgent
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
from sumo_rl.util.scenario import Scenario

class AgentFactory(abc.ABC):
  def __init__(self, env: SumoEnvironment, scenario: Scenario, recycle: bool = False) -> None:
    self.env = env
    self.scenario = scenario
    self.recycle: bool = recycle
  
  @abc.abstractmethod
  def agent_by_assignments(self, assignments: dict[str, list[str]]) -> list[Agent]:
    """Abstract function to create agents for given assignments"""
    pass
  
  @abc.abstractmethod
  def agent_per_ts_shape(self) -> list[Agent]:
    """Abstract function to create agents with traffic signals divided by shape

    If this functionality isn't supported, then the Subclass should define this function to raise a TypeError
    """
    pass
  
  @abc.abstractmethod
  def agent_per_ts(self) -> list[Agent]:
    """Abstract function to create agents, one per traffic signal"""
    pass

  @abc.abstractmethod
  def agent(self, agent_id: str, controlled_entities: dict[str, TrafficSignal], *pargs, **kargs) -> Agent:
    """Abstract function to create an agent"""
    pass

class FixedAgentFactory(AgentFactory):
  def __init__(self, env: SumoEnvironment, scenario: Scenario, recycle: bool = False) -> None:
    super().__init__(env, scenario, recycle)
  
  def agent_by_assignments(self, assignments: dict[str, list[str]]) -> list[QLAgent]:
    agents = []
    traffic_signals = self.env.traffic_signals
    for agent_id, traffic_signal_ids in assignments.items():
      controlled_entities = {traffic_signal_id: traffic_signals[traffic_signal_id] for traffic_signal_id in traffic_signal_ids}
      agents.append(self.agent(agent_id, controlled_entities))
    return agents
  
  def agent_per_ts_shape(self) -> list[QLAgent]:
    raise TypeError("FixedAgentFactory doesn't support agent_per_ts_shape")
  
  def agent_per_ts(self) -> list[QLAgent]:
    assignments = {}
    traffic_signals = self.env.traffic_signals
    for traffic_signal_id in traffic_signals.keys():
      assignments[traffic_signal_id] = [traffic_signal_id]
    return self.agent_by_assignments(assignments)

  def agent(self, agent_id: str, controlled_entities: dict[str, TrafficSignal]) -> QLAgent:
    assert len(controlled_entities) > 0
    a_traffic_signal_id = list(controlled_entities)[0]
    action_space = controlled_entities[a_traffic_signal_id].action_space
    agent = FixedAgent(id=agent_id,
                   controlled_entities=controlled_entities,
                   action_space=action_space)
    return agent

class QLAgentFactory(AgentFactory):
  def __init__(self, env: SumoEnvironment, scenario: Scenario, alpha, gamma, initial_epsilon, min_epsilon, decay, recycle: bool = False) -> None:
    super().__init__(env, scenario, recycle)
    self.alpha: float = alpha
    self.gamma: float = gamma
    self.initial_epsilon: float = initial_epsilon
    self.min_epsilon: float = min_epsilon
    self.decay: float = decay

  def hash_traffic_signal_by_shape(self, observation_fn: ObservationFunction, traffic_signal: TrafficSignal) -> str:
    state_space_hash  = observation_fn.hash(traffic_signal)
    action_space_hash = str(traffic_signal.num_green_phases)
    return "SS%s-%sSS" % (state_space_hash, action_space_hash)
  
  def agent_by_assignments(self, assignments: dict[str, list[str]]) -> list[QLAgent]:
    agents = []
    traffic_signals = self.env.traffic_signals
    for agent_id, traffic_signal_ids in assignments.items():
      controlled_entities = {traffic_signal_id: traffic_signals[traffic_signal_id] for traffic_signal_id in traffic_signal_ids}
      agents.append(self.agent(agent_id, self.env.observation_fn, self.env.reward_fn, controlled_entities))
    return agents
  
  def agent_per_ts_shape(self) -> list[QLAgent]:
    assignments = {}
    traffic_signals = self.env.traffic_signals
    for traffic_signal_id, traffic_signal in traffic_signals.items():
      hash = self.hash_traffic_signal_by_shape(self.env.observation_fn, traffic_signal)
      if hash not in assignments:
        assignments[hash] = []
      assignments[hash].append(traffic_signal_id)
    assignments = {"-".join(ids):ids for ids in assignments.values()}
    return self.agent_by_assignments(assignments)
  
  def agent_per_ts(self) -> list[QLAgent]:
    assignments = {}
    traffic_signals = self.env.traffic_signals
    for traffic_signal_id in traffic_signals.keys():
      assignments[traffic_signal_id] = [traffic_signal_id]
    return self.agent_by_assignments(assignments)

  def agent(self, agent_id: str,
                  observation_fn: ObservationFunction,
                  reward_fn: RewardFunction,
                  controlled_entities: dict[str, TrafficSignal]) -> QLAgent:
    assert len(controlled_entities) > 0
    a_traffic_signal_id = list(controlled_entities)[0]
    action_space = controlled_entities[a_traffic_signal_id].action_space
    state_space = observation_fn.observation_space(controlled_entities[a_traffic_signal_id])
    agent = QLAgent(id=agent_id,
                   observation_fn=observation_fn,
                   reward_fn=reward_fn,
                   controlled_entities=controlled_entities,
                   state_space=state_space,
                   action_space=action_space,
                   alpha=self.alpha,
                   gamma=self.gamma,
                   exploration_strategy=EpsilonGreedy(initial_epsilon=self.initial_epsilon,
                                                      min_epsilon=self.min_epsilon,
                                                      decay=self.decay))
    if self.recycle:
      agent_memory_file = self.scenario.agents_file(None, None, agent_id)
      if os.path.exists(agent_memory_file):
        print("recycle agent %s" % agent_memory_file)
        agent.deserialize(agent_memory_file)
      else:
        print("rebuilding agent %s" % agent_memory_file)
    return agent
