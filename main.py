import os
import sys
import argparse
import sumo_rl.util.utils as utils
import sumo_rl.environment.factories as factories

if "SUMO_HOME" in os.environ:
  tools = os.path.join(os.environ["SUMO_HOME"], "tools")
  sys.path.append(tools)
else:
  sys.exit("Please declare the environment variable 'SUMO_HOME'")

RECICLE=True

if __name__ == "__main__":
  cli = argparse.ArgumentParser(sys.argv[0])
  utils.Scenario.add_scenario_selection(cli)
  cli.add_argument('-f', '--fixed', action="store_true", default=False)
  cli.add_argument('-r', '--recycle', action="store_true", default=False)
  cli_args = cli.parse_args(sys.argv[1:])
  scenario = utils.Scenario(cli_args.scenario)

  env = scenario.new_sumo_environment(cli_args.fixed)
  agent_factory = factories.AgentFactory(
    scenario.config.agent.alpha,
    scenario.config.agent.gamma,
    scenario.config.agent.initial_epsilon,
    scenario.config.agent.min_epsilon,
    scenario.config.agent.decay
  )
  agents = agent_factory.ql_agent_per_ts(env)
  print(agents)
  env.close()
