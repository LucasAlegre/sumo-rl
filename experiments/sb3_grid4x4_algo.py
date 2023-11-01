import os
import shutil
import subprocess

import numpy as np
import supersuit as ss
import traci
from pyvirtualdisplay.smartdisplay import SmartDisplay
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import trange
import matplotlib.pyplot as plt

import sumo_rl

os.environ['LIBSUMO_AS_TRACI'] = "1"
del os.environ['LIBSUMO_AS_TRACI']

RENDER_MODE = os.environ.get("RENDER_MODE", "human")
USE_GUI = os.environ.get("USE_GUI", "True").lower() == "true"
RESOLUTION = (3200, 1800)


def grid4x4_ppo():
    env = sumo_rl.grid4x4(use_gui=USE_GUI, out_csv_name="outputs/grid4x4/ppo_test", virtual_display=RESOLUTION, render_mode=RENDER_MODE)

    max_time = env.unwrapped.env.sim_max_time
    delta_time = env.unwrapped.env.delta_time

    print("Environment created")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        tensorboard_log="./logs/grid4x4/ppo_test",
    )

    print("Starting training")
    model.learn(total_timesteps=50000)

    print("Saving model")
    model.save("./model/grid4x4_ppo")

    # del model  # delete trained model to demonstrate loading
    #
    # print("Loading model")
    # model = PPO.load("./model/grid4x4_ppo", env=env)

    print("Evaluating model")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(mean_reward)
    print(std_reward)

    # Maximum number of steps before reset, +1 because I'm scared of OBOE
    print("Starting rendering")
    num_steps = (max_time // delta_time) + 1

    obs = env.reset()
    for t in trange(num_steps):
        actions, _ = model.predict(obs, state=None, deterministic=False)
        obs, reward, done, info = env.step(actions)
        env.render()

    env.close()


def grid4x4_dqn():
    env = sumo_rl.grid4x4(use_gui=USE_GUI, out_csv_name="outputs/grid4x4/dqn_test", virtual_display=RESOLUTION, render_mode=RENDER_MODE)

    max_time = env.unwrapped.env.sim_max_time
    delta_time = env.unwrapped.env.delta_time

    print("Environment created")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    model = DQN(
        "MlpPolicy",
        env=env,
        learning_rate=1e-3,
        learning_starts=0,
        buffer_size=50000,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.05,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./logs/grid4x4/dqn_test",
    )

    print("Starting training")
    model.learn(total_timesteps=50000)

    print("Saving model")
    model.save("./model/grid4x4_dqn")

    # del model  # delete trained model to demonstrate loading
    #
    # print("Loading model")
    # model = PPO.load("./model/grid4x4_dqn", env=env)

    print("Evaluating model")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(mean_reward)
    print(std_reward)

    # Maximum number of steps before reset, +1 because I'm scared of OBOE
    print("Starting rendering")
    num_steps = (max_time // delta_time) + 1

    obs = env.reset()
    for t in trange(num_steps):
        actions, _ = model.predict(obs, state=None, deterministic=False)
        obs, reward, done, info = env.step(actions)
        env.render()

    env.close()


def grid4x4_a2c():
    pass


def grid4x4_trpo():
    pass


if __name__ == "__main__":
    grid4x4_ppo()
    # grid4x4_dqn()
    # grid4x4_a2c()
    # grid4x4_trpo()
