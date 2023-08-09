import os
import shutil
import subprocess

import numpy as np
import supersuit as ss
import traci
from pyvirtualdisplay.smartdisplay import SmartDisplay
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import trange

import sumo_rl

if __name__ == "__main__":
    RESOLUTION = (3200, 1800)

    env = sumo_rl.grid4x4(use_gui=True, out_csv_name="outputs/grid4x4/ppo_test", virtual_display=RESOLUTION, render_mode="rgb_array")

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

    print("Training finished. Starting evaluation")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(mean_reward)
    print(std_reward)

    # Maximum number of steps before reset, +1 because I'm scared of OBOE
    print("Starting rendering")
    num_steps = (max_time // delta_time) + 1

    obs = env.reset()

    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    # img = disp.grab()
    # img.save(f"temp/img0.jpg")

    img = env.render()
    for t in trange(num_steps):
        actions, _ = model.predict(obs, state=None, deterministic=False)
        obs, reward, done, info = env.step(actions)
        img = env.render()
        img.save(f"temp/img{t}.jpg")

    subprocess.run(["ffmpeg", "-y", "-framerate", "5", "-i", "temp/img%d.jpg", "output.mp4"])

    print("All done, cleaning up")
    shutil.rmtree("temp")
    env.close()

# 测试说明：2023-07-17--2023-08-08
# 1. 本次测试时间为2023-07-17--2023-08-08，在MacOS M1 Ventura 13.4.1 和 Ubuntu 22.04 LTS (GPU 4090)上测试通过。
# 2. 各主要组件的版本如下：
#   - python 3.10.0
#   - pytorch 2.0.1
#   - stable-baselines3 2.0.0a13
#   - pettingzoo 1.23.1
#   - supersuit 3.9.0
#   - sumo-rl 1.4.3
#   - sumo 1.18.0
#   - traci 1.18.0
#   - gymnasium 0.28.1
#   - gym 0.26.2
#   - ray 2.5.0
#   - ray[rllib] 2.5.0
# 3. 必须使用gui模式，否则无法启动服务，客户端无法连接，导致connection refused错误
# 4. 关闭LIBSUMO_AS_TRACI环境变量，使用client-server模式，启动sumo服务,使用traci客户端连接，采用服务间的通信，而非进程间通信。
# 5. experiments里的所有案例都运行成功，画图成功。
# 6. 所有的测试结果都没有分析，所有的超参数都没有进行优化。
# 7. 后期计划
