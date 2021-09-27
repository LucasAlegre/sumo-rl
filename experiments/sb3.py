from stable_baselines3 import PPO
from sumo_rl import SumoEnvironmentPZ, make_env
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from array2gif import write_gif

n_evaluations = 20
n_agents = 2
n_envs = 4
n_timesteps = 8000000

# n agents, n timesteps, docs, make, PZ import in test file
# The main class SumoEnvironment inherits MultiAgentEnv from RLlib.

base_env = make_env(net_file='nets/4x4-Lucas/4x4.net.xml',
                    route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                    out_csv_name='outputs/4x4grid/test',
                    use_gui=False,
                    num_seconds=80000)

env = base_env.copy().parallel_env()
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class='stable_baselines3')
env = VecMonitor(env)

eval_env = base_env.copy().parallel_env()
eval_env = ss.frame_stack_v1(eval_env, 3)
eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
eval_env = ss.concat_vec_envs_v0(eval_env, 1, num_cpus=1, base_class='stable_baselines3')
eval_env = VecMonitor(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs*n_agents), 1)

model = PPO("MlpPolicy", env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=eval_freq, deterministic=True, render=False)
model.learn(total_timesteps=n_timesteps, callback=eval_callback)

model = PPO.load("./logs/best_model")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(mean_reward)
print(std_reward)

render_env = base_env.copy().parallel_env()
render_env = ss.color_reduction_v0(render_env, mode='B')
render_env = ss.resize_v0(render_env, x_size=84, y_size=84)
render_env = ss.frame_stack_v1(render_env, 3)

obs_list = []
i = 0
render_env.reset()


while True:
    for agent in render_env.agent_iter():
        observation, _, done, _ = render_env.last()
        action = model.predict(observation, deterministic=True)[0] if not done else None

        render_env.step(action)
        i += 1
        if i % (len(render_env.possible_agents)) == 0:
            obs_list.append(np.transpose(render_env.render(mode='rgb_array'), axes=(1, 0, 2)))
    render_env.close()
    break

print('Writing gif')
write_gif(obs_list, 'kaz.gif', fps=15)
