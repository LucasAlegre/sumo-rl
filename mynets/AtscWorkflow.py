import os
import sys

sys.path.append('..')

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.dqn import DQN
import mysumo.envs  # 确保自定义环境被注册

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from mysumo.envs.sumo_env import SumoEnv

if __name__ == "__main__":
    env = SumoEnv(
        net_file="my-intersection/my-intersection.net.xml",
        route_file="my-intersection/my-intersection.rou.xml",
        out_csv_name="out/wf-my-intersection-dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=1000,  # 仿真秒，最大值20000
        # render_mode='human', # 'rgb_array':This system has no OpenGL support.
    )

    # env = RecordEpisodeStatistics(env)
    # video_recorder = RecordVideo(env, video_folder='recording', name_prefix="sumo-env-dqn")
    # 异常：last video not closed? 录制视频不成功。
    env = Monitor(env, "monitor/SumoEnv-v0")
    env = DummyVecEnv([lambda: env])

    # 创建算法模型实例，DQN, 试用PPO,A2C, SAC等替换
    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=1000,  # 目标网络更新时间间隔，1000仿真秒
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        tensorboard_log="./tensorboard/wf-my-intersection-dqn",
        verbose=1,
    )

    # 评测模型
    print("evaluate policy====训练前，评测模型的收敛指标")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(mean_reward, std_reward)

    print("train model=====训练模型，总时间步，进度条")
    model.learn(total_timesteps=100000, progress_bar=True)  # 训练总时间步，100000
    print("save model=====保存训练模型")
    model.save("model/wf-my-intersection-dqn")
    print("load model=====加载训练模型")
    model.load("model/wf-my-intersection-dqn")

    # 评测模型
    print("evaluate policy====训练后，评测模型的收敛指标")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(mean_reward, std_reward)

    print("predict====使用模型进行预测")
    env = model.get_env()
    obs = env.reset()
    for i in range(10):
        action, state = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        print("\n======", i, "======")
        print(" obs:", i, obs)
        print(" reward:", reward)
        print(" dones:", dones)
        print(" info:", info)
        env.render()
    env.close()
    del model
