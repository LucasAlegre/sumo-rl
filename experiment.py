import os
import sys
import pickle
import pandas
import sumo_rl.util.utils as utils
import argparse

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
  for run in range(scenario.config.training.runs):
    initial_states = env.reset()
    ql_agents = {}
    if not cli_args.fixed:
      for ts in env.ts_ids:
        if cli_args.recycle:
          ql_agents[ts] = scenario.load_or_new_agent(env, run, ts, initial_states[ts])
        else:
          ql_agents[ts] = scenario.new_agent(env, ts, initial_states[ts])

    for episode in range(scenario.config.training.episodes):
      if episode != 0:
        env.sumo_seed = int(env.sumo_seed) + 1
        initial_states = env.reset()
        if not cli_args.fixed:
          for ts in initial_states.keys():
            ql_agents[ts].state = env.encode(initial_states[ts], ts)

      done = {"__all__": False}
      while not done["__all__"]:
        if not cli_args.fixed:
          actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
          try:
            s, r, done, info = env.step(action=actions)
            for agent_id in s.keys():
              ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
          except ValueError as err:
            print(err)
            print("actions", actions)
            print("states", {ts: ql_agents[ts].state for ts in ql_agents.keys()})
            print("action_spaces", {ts: ql_agents[ts].action_space for ts in ql_agents.keys()})
            os.abort()
        else:
          s, r, done, info = env.step(action={})

      path = scenario.metrics_file(run, episode)
      pandas.DataFrame(env.metrics).to_csv(path, index=False)
      if not cli_args.fixed:
        for ts, agent in ql_agents.items():
          path = scenario.agents_file(run, episode, ts)
          with open(path, "wb") as file:
            pickle.dump(agent.q_table, file)
    if not cli_args.fixed:
      for ts, agent in ql_agents.items():
        path = scenario.agents_file(run, None, ts)
        with open(path, "wb") as file:
          pickle.dump(agent.q_table, file)
  env.close()
