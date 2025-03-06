import typing
import pandas
import matplotlib.pyplot
import sumo_rl.util.scenario as scenario
import argparse
import sys

def load_metrics(scenario: scenario.Scenario) -> dict[int, dict[int, pandas.DataFrame]]:
  metrics = {}
  for run in range(scenario.config.training.runs):
    metrics[run] = {}
    for episode in range(scenario.config.training.episodes):
      metrics[run][episode] = pandas.read_csv(scenario.metrics_file(run, episode))
  return metrics

def plot_single_metrics(metrics: dict[int, dict[int, pandas.DataFrame]], label: str, retrieve_data: typing.Callable):
  for run in metrics:
    for episode in metrics[run]:
      df = metrics[run][episode]
      _ = matplotlib.pyplot.figure(figsize=(20, 10))
      Ys = retrieve_data(df)
      Xs = [_ for _ in range(len(Ys))]
      matplotlib.pyplot.plot(Xs, Ys, marker='o', label=label)
      matplotlib.pyplot.title('Metrics for run %s / episode %s' % (run, episode))
      matplotlib.pyplot.legend()
      matplotlib.pyplot.savefig(scenario.plots_file(label, run, episode))
      matplotlib.pyplot.close()

def plot_summary_metrics(metrics: dict[int, dict[int, pandas.DataFrame]], label: str, retrieve_data: typing.Callable):
  for run in metrics:
    _ = matplotlib.pyplot.figure(figsize=(20, 10))
    Ys = []
    for episode in metrics[run]:
      df = metrics[run][episode]
      Ys += retrieve_data(df)
    Xs = [_ for _ in range(len(Ys))]
    matplotlib.pyplot.plot(Xs, Ys, marker='o', label=label)
    matplotlib.pyplot.title('Metrics for run %s' % (run))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(scenario.plots_file(label, run, None))
    matplotlib.pyplot.close()

def symmetric_smoother(data: list, K: int) -> list:
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
    value = sum(data[lbound:hbound])
    output.append(value)
  return output

def asymmetric_smoother(data: list, K: int) -> list:
  """Smooths data[0:N] by factor K returning output[0:M], with K <= N and M <= N"""
  N = len(data)
  assert K <= N
  output = []
  for i in range(N):
    lbound = i
    hbound = i + K
    if hbound > N:
      break
    value = sum(data[lbound:hbound])
    output.append(value)
  return output

if __name__ == "__main__":
  cli = argparse.ArgumentParser(sys.argv[0])
  scenario.Scenario.add_scenario_selection(cli)
  cli_args = cli.parse_args(sys.argv[1:])
  scenario = scenario.Scenario(cli_args.scenario)
  metrics = load_metrics(scenario)

  system_total_running = lambda df: list(df['system_total_running'])
  system_total_stopped = lambda df: list(df['system_total_stopped'])
  system_total_waiting_time = lambda df: list(df['system_total_waiting_time'])
  system_mean_waiting_time = lambda df: list(df['system_mean_waiting_time'])
  vehicles_number = lambda df: list(df['system_total_waiting_time'] / df['system_mean_waiting_time'])

  ss_system_total_running =       lambda df: symmetric_smoother(system_total_running(df), 50)
  ss_system_total_stopped =       lambda df: symmetric_smoother(system_total_stopped(df), 50)
  ss_system_total_waiting_time =  lambda df: symmetric_smoother(system_total_waiting_time(df), 50)
  ss_system_mean_waiting_time =   lambda df: symmetric_smoother(system_mean_waiting_time(df), 50)
  ss_vehicles_number =            lambda df: symmetric_smoother(vehicles_number(df), 50)
 
  as_system_total_running =       lambda df: asymmetric_smoother(system_total_running(df), 50)
  as_system_total_stopped =       lambda df: asymmetric_smoother(system_total_stopped(df), 50)
  as_system_total_waiting_time =  lambda df: asymmetric_smoother(system_total_waiting_time(df), 50)
  as_system_mean_waiting_time =   lambda df: asymmetric_smoother(system_mean_waiting_time(df), 50)
  as_vehicles_number =            lambda df: asymmetric_smoother(vehicles_number(df), 50)
  
  quests = [
    ('system_mean_waiting_time', system_mean_waiting_time),
    ('ss_system_mean_waiting_time', ss_system_mean_waiting_time),
    ('as_system_mean_waiting_time', as_system_mean_waiting_time),
  ]

  for (label, retriever) in quests:
    plot_single_metrics(metrics, label, retriever)
    plot_summary_metrics(metrics, label, retriever)
