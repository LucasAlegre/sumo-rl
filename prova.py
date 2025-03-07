import os
import sys
import argparse

import sumo_rl.environment.env
import sumo_rl.util.scenario
import sumo_rl.preprocessing.factories
import sumo_rl.environment.observations
import sumo_rl.environment.rewards
import sumo_rl.preprocessing.graphs
import sumo_rl.preprocessing.partitions
import sumo_rl.agents

if "SUMO_HOME" in os.environ:
  tools = os.path.join(os.environ["SUMO_HOME"], "tools")
  sys.path.append(tools)
else:
  sys.exit("Please declare the environment variable 'SUMO_HOME'")

if __name__ == "__main__":
  cli = argparse.ArgumentParser(sys.argv[0])
  sumo_rl.util.scenario.Scenario.add_scenario_selection(cli)
  cli_args = cli.parse_args(sys.argv[1:])
  scenario = sumo_rl.util.scenario.Scenario(cli_args.scenario)
  env = scenario.new_sumo_environment(sumo_rl.environment.observations.DefaultObservationFunction(), sumo_rl.environment.rewards.AverageSpeedRewardFunction())

  by_size_partition = sumo_rl.preprocessing.partitions.ActionStateSizePartition.Build(env)
  by_space_partition = sumo_rl.preprocessing.partitions.ActionStateSpacePartition.Build(env)
  factory = sumo_rl.preprocessing.factories.QLAgentFactory(env, scenario, 
                                                           scenario.config.agent.alpha,
                                                           scenario.config.agent.gamma,
                                                           scenario.config.agent.initial_epsilon,
                                                           scenario.config.agent.min_epsilon,
                                                           scenario.config.agent.decay)
  print("By Size")
  for agent in factory.agent_by_assignments(by_size_partition.data):
    print(agent)
  print("By Space")
  for agent in factory.agent_by_assignments(by_space_partition.data):
    print(agent)
  env.close()
