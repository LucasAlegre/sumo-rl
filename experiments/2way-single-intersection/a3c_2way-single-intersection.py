import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
from sumo_rl.util.gen_route import write_route_file
import traci

#from stable_baselines3.common.vec_env import VecMonitor
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, DQN
import ray
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy

if __name__ == '__main__':

    write_route_file('nets/single-intersection/single-intersection-gen.rou.xml', 400000, 100000)

    env = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                        route_file='nets/2way-single-intersection/single-intersection-gen.rou.xml',
                                        out_csv_name='outputs/2way-single-intersection/a2c',
                                        single_agent=True,
                                        use_gui=False,
                                        num_seconds=100000,
                                        min_green=5)])
    model = A3C("MlpPolicy", 
				env, 
				verbose=1, 
				learning_rate=0.001)
    model.learn(total_timesteps=100000)
    model.save('a3c-2way-single-intersection')
