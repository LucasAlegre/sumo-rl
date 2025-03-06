import os
import sys
import argparse
import pandas
import sumo_rl.util.scenario
import sumo_rl.environment.factories
import sumo_rl.agents

if "SUMO_HOME" in os.environ:
  tools = os.path.join(os.environ["SUMO_HOME"], "tools")
  sys.path.append(tools)
else:
  sys.exit("Please declare the environment variable 'SUMO_HOME'")

if __name__ == "__main__":
  cli = argparse.ArgumentParser(sys.argv[0])
  sumo_rl.util.scenario.Scenario.add_scenario_selection(cli)
  cli.add_argument('-f', '--fixed', action="store_true", default=False)
  cli.add_argument('-r', '--recycle', action="store_true", default=False)
  cli.add_argument('-a', '--aggregate', action="store_true", default=False)
  cli_args = cli.parse_args(sys.argv[1:])
  scenario = sumo_rl.util.scenario.Scenario(cli_args.scenario)

  env = scenario.new_sumo_environment()
  agent_factory: sumo_rl.environment.factories.AgentFactory
  if cli_args.fixed:
    agent_factory = sumo_rl.environment.factories.FixedAgentFactory(
      env, scenario,
      recycle=cli_args.recycle
    )
  else:
    agent_factory = sumo_rl.environment.factories.QLAgentFactory(
      env, scenario,
      scenario.config.agent.alpha,
      scenario.config.agent.gamma,
      scenario.config.agent.initial_epsilon,
      scenario.config.agent.min_epsilon,
      scenario.config.agent.decay,
      recycle=cli_args.recycle
    )
  agents: list[sumo_rl.agents.Agent]
  if cli_args.aggregate:
    agents = agent_factory.agent_per_ts_shape()
  else:
    agents = agent_factory.agent_per_ts()

  env.sumo_seed = scenario.config.sumo.sumo_seed
  for run in range(scenario.config.training.runs):
    for episode in range(scenario.config.training.episodes):
      conn = env.reset()
      for agent in agents:
        agent.reset(conn)
      
      while not env.done():
        actions = {}
        for agent in agents:
          agent.observe()
        for agent in agents:
          actions.update(agent.act())
        env.step(action=actions)
        env.compute_info()
        if not cli_args.fixed:
          for agent in agents:
            agent.learn()

      # Serialize Metrics
      path = scenario.metrics_file(run, episode)
      pandas.DataFrame(env.metrics).to_csv(path, index=False)
      # Serialize Agents
      if not cli_args.fixed:
        for agent in agents:
          path = scenario.agents_file(run, episode, agent.id)
          agent.serialize(path)

      env.sumo_seed += 1
    # Serialize Agents
    if not cli_args.fixed:
      for agent in agents:
        path = scenario.agents_file(run, None, agent.id)
        agent.serialize(path)
  # Serialize Agents
  if not cli_args.fixed:
    for agent in agents:
      path = scenario.agents_file(None, None, agent.id)
      agent.serialize(path)
  env.close()
