from mysumo import SumoEnv

if __name__ == "__main__":
    env = SumoEnv(net_file="/tmp/net.xml", route_file="/tmp/route.xml")
    print("env:", env)
