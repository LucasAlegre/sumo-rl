from pettingzoo.test import api_test
from sumo_rl import SumoEnvironmentPZ

env = SumoEnvironmentPZ(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                    route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                    out_csv_name='outputs/4x4grid/a3c',
                                                    use_gui=False,
                                                    num_seconds=80000,
                                                    max_depart_delay=0)

api_test(env)
