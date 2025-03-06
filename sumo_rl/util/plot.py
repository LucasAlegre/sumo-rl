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
      matplotlib.pyplot.plot(df['step'], retrieve_data(df), marker='o', label=label)
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

if __name__ == "__main__":
  cli = argparse.ArgumentParser(sys.argv[0])
  scenario.Scenario.add_scenario_selection(cli)
  cli_args = cli.parse_args(sys.argv[1:])
  scenario = scenario.Scenario(cli_args.scenario)
  metrics = load_metrics(scenario)
  plot_single_metrics(metrics, 'system_total_running', lambda df: list(df['system_total_running']))
  plot_single_metrics(metrics, 'system_total_stopped', lambda df: list(df['system_total_stopped']))
  plot_single_metrics(metrics, 'system_total_waiting_time', lambda df: list(df['system_total_waiting_time']))
  plot_single_metrics(metrics, 'system_mean_waiting_time', lambda df: list(df['system_mean_waiting_time']))
  plot_single_metrics(metrics, 'vehicles_number', lambda df: list(df['system_total_waiting_time'] / df['system_mean_waiting_time']))
  plot_summary_metrics(metrics, 'system_total_running', lambda df: list(df['system_total_running']))
  plot_summary_metrics(metrics, 'system_total_stopped', lambda df: list(df['system_total_stopped']))
  plot_summary_metrics(metrics, 'system_total_waiting_time', lambda df: list(df['system_total_waiting_time']))
  plot_summary_metrics(metrics, 'system_mean_waiting_time', lambda df: list(df['system_mean_waiting_time']))
  plot_summary_metrics(metrics, 'vehicles_number', lambda df: list(df['system_total_waiting_time'] / df['system_mean_waiting_time']))
