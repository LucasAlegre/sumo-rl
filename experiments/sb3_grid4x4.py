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
import matplotlib.pyplot as plt

import sumo_rl

os.environ['LIBSUMO_AS_TRACI'] = "1"
del os.environ['LIBSUMO_AS_TRACI']

RENDER_MODE = os.environ.get("RENDER_MODE", "human")
USE_GUI = os.environ.get("USE_GUI", "True").lower() == "true"

# display = SmartDisplay(visible=False, size=(800, 600))
# display.start()

if __name__ == "__main__":
    RESOLUTION = (3200, 1800)

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
    model.save("./model/ppo_grid4x4")

    del model  # delete trained model to demonstrate loading
    #
    print("Loading model")
    env = sumo_rl.grid4x4(use_gui=True, out_csv_name="outputs/grid4x4/ppo_test", virtual_display=RESOLUTION, render_mode="rgb_array")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)
    model = PPO.load("./model/ppo_grid4x4", env=env)

    print("Evaluating model")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(mean_reward)
    print(std_reward)

    # Maximum number of steps before reset, +1 because I'm scared of OBOE
    print("Starting rendering")
    num_steps = (max_time // delta_time) + 1


    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    # img = disp.grab()
    # img.save(f"temp/img0.jpg")

    obs = env.reset()
    for t in trange(num_steps):
        actions, _ = model.predict(obs, state=None, deterministic=False)
        obs, reward, done, info = env.step(actions)
        img = env.render(mode="rgb_array")
        img.save(f"temp/img{t}.jpg")

    subprocess.run(["ffmpeg", "-y", "-framerate", "5", "-i", "temp/img%d.jpg", "output.mp4"])

    print("All done, cleaning up")
    shutil.rmtree("temp")
    env.close()

# 测试说明：2023-08-08
# 1. 在MacOS M1 Ventura 13.4.1 和 Ubuntu 22.04 LTS (GPU 4090)上测试通过。
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
# 3. 必须使用gui模式(use_gui=True)，否则无法启动服务，客户端无法连接，导致connection refused错误
# 4. 关闭LIBSUMO_AS_TRACI环境变量（unset LIBSUMO_AS_TRACI），使用client-server模式，启动sumo服务,使用traci客户端连接，采用服务间的通信，而非进程间通信。
# 5. 无法使用render_mode="rgb_array"返回数组，只能使用render_mode="human"返回图像。

# 测试说明：2021-10-30
# 检查环境，见列表 libs-mac/ubuntu-2023-10-30.txt
# 运行程序, 报错：Too many parameters for <class 'gymnasium.core.Wrapper'>; actual 4, expected 2
# 1, create -n sumoai-sb3-grid4x4 python=3.10
#  pip install pytorch==2.0.1 torchvision==2.0.1 torchaudio==2.0.1
#  pip install stable-baselines3==2.0.0a13
#  pip install pettingzoo==1.23.1
#  pip install supersuit==3.9.0
#  pip install sumolib==1.18.0
#  pip install traci==1.18.0
#  pip install gymnasium==0.28.1
#  pip install ray[rllib]==2.5.0
#  pip install pyvirtualdisplay
# 2, 运行时报错，显示Nvidia driver/library version mismatch. 这是由于重装pytorch导致的，需要重新安装Nvidia驱动。
#  按官网指导安装了新版本，driver 545.23.06, cuda toolkit 12.3。
# 3, ubuntu里按上面要求重装后，运行成功。
# 4, macos里，重装torch等libs，运行时出现No OpenGL support错误，关闭SmartDisplay后，开启sumo-gui模式运行成功。
