import sumo_rl.environment.env

class Graph:
  def __init__(self) -> None:
    self.data: dict[str, set[str]] = {}

  def add_asymmetric_edge(self, from_ID: str, to_ID: str):
    if from_ID not in self.data:
      self.data[from_ID] = set({})
    if to_ID not in self.data[from_ID]:
      self.data[from_ID].add(to_ID)

  def add_symmetric_edge(self, from_ID: str, to_ID: str):
    self.add_asymmetric_edge(from_ID, to_ID)
    self.add_asymmetric_edge(to_ID, from_ID)

class AdiacencyGraph(Graph):
  @staticmethod
  def Build(env: sumo_rl.environment.env.SumoEnvironment):
    graph = __class__()
    traffic_signals = set(env.traffic_signals.keys())
    edges = env.sumo.edge.getIDList()
    for edge_ID in edges:
      from_junction = env.sumo.edge.getFromJunction(edge_ID)
      to_junction = env.sumo.edge.getToJunction(edge_ID)
      if from_junction in traffic_signals:
        if to_junction in traffic_signals:
          graph.add_symmetric_edge(from_junction, to_junction)
    return graph
