    
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
from stable_baselines3 import PPO


if __name__ == '__main__':

    write_route_file('nets/2way-single-intersection/single-intersection-gen.rou.xml', 400000, 100000)

    env = SubprocVecEnv([lambda: SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                                        route_file='nets/2way-single-intersection/single-intersection.rou.xml',
                                        out_csv_name='outputs/2way-single-intersection/ppo',
                                        single_agent=True,
                                        use_gui=False,
                                        num_seconds=100000,
                                        min_green=5)])
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                gamma=0.95,
                n_steps=256,
                ent_coef=0.0905168,
                learning_rate=0.00062211,
                vf_coef=0.042202,
                max_grad_norm=0.9,
                gae_lambda=0.99,
                n_epochs=1,
                clip_range=0.3,
                batch_size=256,
                #tensorboard_log="./logs/grid4x4/ppo_test",)
   		tensorboard_log=None)
    model.learn(total_timesteps=100000)
    model.save('ppo-2way-single-intersection.pkl')
