from __future__ import annotations

import typing
import pandas
import matplotlib.pyplot
import sumo_rl.util.scenario
import argparse
import sys
import enum

class Datastore:
  class Mode(enum.Enum):
    TRAINING='training'
    EVALUATION='evaluation'

  def __init__(self, scenario: sumo_rl.util.scenario.Scenario, mode: Datastore.Mode) -> None:
    self.scenario = scenario
    self.mode = mode
    self.runs, self.episodes = self._load_runs_and_episodes()
    self.metrics: dict[int, dict[int, pandas.DataFrame]] = self._load_metrics()

  def _load_runs_and_episodes(self):
    if self.mode == Datastore.Mode.EVALUATION:
      return self.scenario.config.evaluation.runs, self.scenario.config.evaluation.episodes
    elif self.mode == Datastore.Mode.TRAINING:
      return self.scenario.config.training.runs, self.scenario.config.training.episodes
    else:
      raise ValueError(self.mode)

  def _load_metrics(self) -> dict[int, dict[int, pandas.DataFrame]]:
    metrics = {}
    for run in range(self.runs):
      metrics[run] = {}
      for episode in range(self.episodes):
        metrics[run][episode] = pandas.read_csv(self.metrics_file(run, episode))
    return metrics

  def metrics_file(self, run, episode) -> str:
    if self.mode == Datastore.Mode.EVALUATION:
      return self.scenario.evaluation_metrics_file(run, episode)
    elif self.mode == Datastore.Mode.TRAINING:
      return self.scenario.training_metrics_file(run, episode)
    else:
      raise ValueError(self.mode)

  def plots_file(self, label, run, episode) -> str:
    if self.mode == Datastore.Mode.EVALUATION:
      return self.scenario.evaluation_plots_file(label, run, episode)
    elif self.mode == Datastore.Mode.TRAINING:
      return self.scenario.training_plots_file(label, run, episode)
    else:
      raise ValueError(self.mode)

  def __repr__(self) -> str:
    return "Datastore(%s)" % (self.mode,)

class Smoother:
  @staticmethod
  def Symmetric(data: list, K: int) -> list:
    """Smooths data[0:N] by factor K returning output[0:N], where K <= N"""
    N = len(data)
    assert K <= N
    half_K = K // 2
    output = []
    for i in range(N):
      lbound = i - half_K
      if lbound < 0:
        lbound = 0
      hbound = i + half_K
      if hbound > N:
        hbound = N
      value = sum(data[lbound:hbound])/len(data[lbound:hbound])
      output.append(value)
    return output
  
  @staticmethod
  def Asymmetric(data: list, K: int) -> list:
    """Smooths data[0:N] by factor K returning output[0:M], with K <= N and M <= N"""
    N = len(data)
    assert K <= N
    output = []
    for i in range(N):
      lbound = i
      hbound = i + K
      if hbound > N:
        break
      value = sum(data[lbound:hbound])/len(data[lbound:hbound])
      output.append(value)
    return output

  @staticmethod
  def Apply(retriever: Retriever, symmetric: bool) -> Retriever:
    if symmetric:
      return lambda df: Smoother.Symmetric(retriever(df), 50)
    return lambda df: Smoother.Asymmetric(retriever(df), 50)

class Plotter:
  @staticmethod
  def Single(datastore: Datastore, label: str, retrieve_data: Retriever):
    for run in datastore.metrics:
      for episode in datastore.metrics[run]:
        df = datastore.metrics[run][episode]
        _ = matplotlib.pyplot.figure(figsize=(20, 10))
        Ys = retrieve_data(df)
        Xs = [_ for _ in range(len(Ys))]
        matplotlib.pyplot.plot(Xs, Ys, marker='o', label=label)
        matplotlib.pyplot.title('Metrics for run %s / episode %s' % (run, episode))
        matplotlib.pyplot.legend()
        matplotlib.pyplot.savefig(datastore.plots_file(label, run, episode))
        matplotlib.pyplot.close()
  
  @staticmethod
  def Summary(datastore: Datastore, label: str, retrieve_data: typing.Callable):
    for run in datastore.metrics:
      _ = matplotlib.pyplot.figure(figsize=(20, 10))
      Ys = []
      for episode in datastore.metrics[run]:
        df = datastore.metrics[run][episode]
        Ys += retrieve_data(df)
      Xs = [_ for _ in range(len(Ys))]
      matplotlib.pyplot.plot(Xs, Ys, marker='o', label=label)
      matplotlib.pyplot.title('Metrics for run %s' % (run))
      matplotlib.pyplot.legend()
      matplotlib.pyplot.savefig(datastore.plots_file(label, run, None))
      matplotlib.pyplot.close()


Retriever = typing.Callable[[pandas.DataFrame], list]

class Plot:
  def __init__(self, datastore: Datastore, label: str, retrieve_data: Retriever, single: bool, summary: bool):
    self.datastore = datastore
    self.label = label
    self.retrieve_data = retrieve_data
    self.single = single
    self.summary = summary

  def plot(self):
    if self.single:
      Plotter.Single(self.datastore, self.label, self.retrieve_data)
    if self.summary:
      Plotter.Summary(self.datastore, self.label, self.retrieve_data)
    print(self)

  def copy(self):
    return Plot(self.datastore, self.label, self.retrieve_data, self.single, self.summary)

  def __repr__(self) -> str:
    return "Plot(%s, %s)" % (self.datastore, self.label)

class Preprocessor:
  @staticmethod
  def InitialSubset(datastore: Datastore) -> list[Plot]:
    plots = []
    plots.append(Plot(datastore, "total_running", lambda df: list(df['total_running']), True, True))
    plots.append(Plot(datastore, "total_stopped", lambda df: list(df['total_stopped']), True, True))
    plots.append(Plot(datastore, "total_waiting_time", lambda df: list(df['total_waiting_time']), True, True))
    plots.append(Plot(datastore, "mean_waiting_time", lambda df: list(df['mean_waiting_time']), True, True))
    plots.append(Plot(datastore, "vehicles_number", lambda df: list(df['total_waiting_time'] / df['mean_waiting_time']), True, True))
    return plots

  @staticmethod
  def InitialSet(scenario: sumo_rl.util.scenario.Scenario) -> list[Plot]:
    training_datastore = Datastore(scenario, Datastore.Mode.TRAINING)
    evaluation_datastore = Datastore(scenario, Datastore.Mode.EVALUATION)

    plots = []
    plots += Preprocessor.InitialSubset(training_datastore)
    plots += Preprocessor.InitialSubset(evaluation_datastore)
    return plots

  @staticmethod
  def ApplySmoothing(plots: list[Plot]) -> list[Plot]:
    output = []
    for plot in plots:
      output.append(plot)
      with_asym_smoothing = plot.copy()
      with_asym_smoothing.label = with_asym_smoothing.label + '-AS'
      with_asym_smoothing.retrieve_data = Smoother.Apply(with_asym_smoothing.retrieve_data, False)
      output.append(with_asym_smoothing)
      with_sym_smoothing = plot.copy()
      with_sym_smoothing.label = with_sym_smoothing.label + '-SS'
      with_sym_smoothing.retrieve_data = Smoother.Apply(with_asym_smoothing.retrieve_data, True)
      output.append(with_sym_smoothing)
    return output

if __name__ == "__main__":
  cli = argparse.ArgumentParser(sys.argv[0])
  sumo_rl.util.scenario.Scenario.add_scenario_selection(cli)
  cli_args = cli.parse_args(sys.argv[1:])
  scenario = sumo_rl.util.scenario.Scenario(cli_args.scenario)

  plots = Preprocessor.ApplySmoothing(Preprocessor.InitialSet(scenario))
  for plot in plots:
    plot.plot()
