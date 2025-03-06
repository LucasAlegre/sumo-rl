import os
import pickle
import yaml

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

class SumoConfig:
  def __init__(self, data: dict):
    self.seconds: int = data['seconds']
    self.min_green: int = data['min_green']
    self.delta_time: int = data['delta_time']
    self.use_gui: bool = data['use_gui']
    self.sumo_seed: int = data['sumo_seed']

class AgentConfig:
  def __init__(self, data: dict):
    self.alpha: float = data['alpha']
    self.gamma: float = data['gamma']
    self.initial_epsilon: float = data['initial_epsilon']
    self.min_epsilon: float = data['min_epsilon']
    self.decay: int = data['decay']

class TrainingConfig:
  def __init__(self, data: dict):
    self.runs: int = data['runs']
    self.episodes: int = data['episodes']

class Config:
  def __init__(self, data: dict):
    self.sumo: SumoConfig = SumoConfig(data['sumo'])
    self.agent: AgentConfig = AgentConfig(data['agent'])
    self.training: TrainingConfig = TrainingConfig(data['training'])
  
  @staticmethod
  def from_file(filepath: str):
    with open(filepath, "r") as file:
      return Config(yaml.load(file, Loader=yaml.Loader))

class Scenario:
  def __init__(self, name: str) -> None:
    self.name = name
    self.config = Config.from_file(self.config_file())

  @staticmethod
  def list_of_scenarios() -> list[str]:
    scenarios: list[str] = []
    for name in os.listdir('./scenarios/'):
      scenarios.append(name)
    return scenarios

  @staticmethod
  def add_scenario_selection(cli):
    scenarios = Scenario.list_of_scenarios()
    assert len(scenarios) > 0
    cli.add_argument('-s', '--scenario', type=str, default=scenarios[0], choices=scenarios, help='Selects the scenario. (defaults to %s)' % scenarios[0])

  def ensure_dir(self, dir: str) -> str:
    if not os.path.exists(dir):
      os.makedirs(dir)
    return dir

  def config_file(self, ) -> str:
    return './scenarios/%s/config.yml' % self.name

  def agents_dir(self, run: int|None, episode: int|None) -> str:
    if episode is None:
      if run is None:
        return self.ensure_dir("./outputs/%s/agents/final" % (self.name))
      return self.ensure_dir("./outputs/%s/agents/%s/final" % (self.name, run))
    assert run is not None
    return self.ensure_dir("./outputs/%s/agents/%s/%s" % (self.name, run, episode))

  def agents_file(self, run: int|None, episode: int|None, agent: int) -> str:
    return "./%s/%s.pickle" % (self.agents_dir(run, episode), agent)

  def metrics_dir(self, run: int) -> str:
    return self.ensure_dir("./outputs/%s/metrics/%s" % (self.name, run))

  def metrics_file(self, run: int, episode: int) -> str:
    return "./%s/%s.csv" % (self.metrics_dir(run), episode)

  def plots_dir(self, label: str, run: int) -> str:
    return self.ensure_dir("./outputs/%s/plots/%s/%s" % (self.name, label, run))

  def plots_file(self, label: str, run: int, episode: int|None) -> str:
    if episode is None:
      return "./%s/summary.png" % (self.plots_dir(label, run))
    return "./%s/%s.png" % (self.plots_dir(label, run), episode)

  def network_file(self) -> str:
    return "./scenarios/%s/network.net.xml" % self.name

  def route_file(self) -> str:
    return "./scenarios/%s/routes.rou.xml" % self.name

  def new_sumo_environment(self) -> SumoEnvironment:
    return SumoEnvironment(
      net_file=self.network_file(),
      route_file=self.route_file(),
      use_gui=self.config.sumo.use_gui,
      num_seconds=self.config.sumo.seconds,
      min_green=self.config.sumo.min_green,
      delta_time=self.config.sumo.delta_time,
      sumo_seed=self.config.sumo.sumo_seed,
      fixed_ts=False,
    )
