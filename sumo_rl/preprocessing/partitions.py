import sumo_rl.environment.env
import sumo_rl.environment.traffic_signal
import sumo_rl.observations
import sumo_rl.environment.traffic_signal

class Partition:
  def __init__(self) -> None:
    self.data: dict[str, set[str]] = {}

  def add(self, partition_ID: str, element: str) -> None:
    if partition_ID not in self.data:
      self.data[partition_ID] = set({})
    self.data[partition_ID].add(element)

class MonadicPartition(Partition):
  """Each traffic signal goes into a separated partition"""
  @staticmethod
  def Hash(traffic_signal: sumo_rl.environment.traffic_signal.TrafficSignal) -> str:
    return traffic_signal.id

  @staticmethod
  def Build(env: sumo_rl.environment.env.SumoEnvironment):
    partition = __class__()
    traffic_signals = env.traffic_signals
    for traffic_signal_ID, traffic_signal in traffic_signals.items():
      partition.add(partition.Hash(traffic_signal), traffic_signal_ID)
    return partition

class ActionStateSizePartition(Partition):
  """Size is measured as lengths of traffic_signal program's state and the number of green phases (= of actions)"""
  @staticmethod
  def Hash(traffic_signal: sumo_rl.environment.traffic_signal.TrafficSignal) -> str:
    return "%s-%s" % (len(traffic_signal.green_phases), len(traffic_signal.green_phases[0].state))

  @staticmethod
  def Build(env: sumo_rl.environment.env.SumoEnvironment):
    partition = __class__()
    traffic_signals = env.traffic_signals
    for traffic_signal_ID, traffic_signal in traffic_signals.items():
      partition.add(partition.Hash(traffic_signal), traffic_signal_ID)
    return partition

class ActionStateSpacePartition(Partition):
  """Shape is measured as size of observation space and size of action space"""
  @staticmethod
  def Hash(observation_fn: sumo_rl.observations.ObservationFunction, traffic_signal: sumo_rl.environment.traffic_signal.TrafficSignal) -> str:
    state_space_hash  = observation_fn.hash(traffic_signal)
    action_space_hash = str(traffic_signal.num_green_phases)
    return "SS%s-%sSS" % (state_space_hash, action_space_hash)

  @staticmethod
  def Build(env: sumo_rl.environment.env.SumoEnvironment):
    partition = __class__()
    traffic_signals = env.traffic_signals
    for traffic_signal_ID, traffic_signal in traffic_signals.items():
      partition.add(partition.Hash(env.observation_fn, traffic_signal), traffic_signal_ID)
    return partition
