import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import sumo_rl


def test_api():
    env = gym.make(
        "sumo-rl-v0",
        num_seconds=100,
        use_gui=False,
        net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
    )
    env.reset()
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


if __name__ == "__main__":
    test_api()
