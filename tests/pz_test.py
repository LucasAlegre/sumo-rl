from pettingzoo.test import api_test
import sumo_rl


def test_api():
    env = sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                   route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                   out_csv_name='outputs/4x4grid/test',
                   use_gui=False,
                   num_seconds=80000)
    api_test(env)
    env.close()

if __name__ == '__main__':
    test_api()
