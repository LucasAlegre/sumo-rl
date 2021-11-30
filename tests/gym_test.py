from sumo_rl import SumoEnvironment
from gym.utils.env_checker import check_env


def test_api():
    env = SumoEnvironment(single_agent=True,
                          num_seconds=100000,
                          net_file='nets/single-intersection/single-intersection.net.xml',
                          route_file='nets/single-intersection/single-intersection.rou.xml')
    env.reset()
    check_env(env)
    env.close()

if __name__ == '__main__':
    test_api()

