import argparse
import os
import sys
from datetime import datetime


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

from sumo_rl import SumoEnvironment
from sumo_rl.util.gen_route import write_route_file


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""SarsaLambda Single-Intersection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="nets/2way-single-intersection/single-intersection-gen.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.01, required=False, help="Epsilon.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=400000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()

    out_csv = "outputs/2way-single-intersection/sarsa_lambda"

    write_route_file("sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml", 400000, 100000)
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        single_agent=True,
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
    )

    for run in range(1, args.runs + 1):
        obs, info = env.reset()
        agent = TrueOnlineSarsaLambda(
            env.observation_space,
            env.action_space,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            fourier_order=7,
            lamb=0.95,
        )

        terminated, truncated = False, False
        if args.fixed:
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step({})
        else:
            while not (terminated or truncated):
                action = agent.act(obs)

                next_obs, r, terminated, truncated, info = env.step(action=action)

                agent.learn(state=obs, action=action, reward=r, next_state=next_obs, done=terminated)

                obs = next_obs

        env.save_csv(out_csv, run)
