from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym


def cartpole_ppo():
    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=4)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("./model/sb3_cartpole_ppo")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("./model/sb3_cartpole_ppo")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones.any():
            obs = vec_env.reset()


def cartpole_dqn():
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="human")
    # 把环境向量化，如果有多个环境写成列表传入DummyVecEnv中，可以用一个线程来执行多个环境，提高训练效率
    env = DummyVecEnv([lambda: env])
    # 定义一个DQN模型，设置其中的各个参数
    model = DQN(
        "MlpPolicy",  # MlpPolicy定义策略网络为MLP网络
        env=env,
        learning_rate=5e-4,
        batch_size=128,
        buffer_size=50000,
        learning_starts=0,
        target_update_interval=250,
        policy_kwargs={"net_arch": [256, 256]},  # 这里代表隐藏层为2层256个节点数的网络
        verbose=1,  # verbose=1代表打印训练信息，如果是0为不打印，2为打印调试信息
        tensorboard_log="./tensorboard/sb3_cartpole_ppo/"  # 训练数据保存目录，可以用tensorboard查看
    )
    # 开始训练
    model.learn(total_timesteps=100000)
    # 策略评估，可以看到倒立摆在平稳运行了
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    # env.close()
    print("mean_reward:", mean_reward, "std_reward:", std_reward)
    # 保存模型到相应的目录
    model.save("./model/sb3_cartpole_dqn")


def cartpole_a2c():
    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=4)

    model = A2C("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("./model/sb3_cartpole_a2c")

    del model  # remove to demonstrate saving and loading

    print("Loading model and predict...")
    model = A2C.load("./model/sb3_cartpole_a2c")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones.all():
            break


def cartpole_sac():
    env = gym.make("CartPole-v1", render_mode="human")

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("./model/sb3_cartpole_sac")

    del model  # remove to demonstrate saving and loading

    model = SAC.load("./model/sb3_cartpole_sac")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()


def pendulum_sac():
    env = gym.make("Pendulum-v1", render_mode="human")

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("./model/sb3_pendulum_sac")

    del model  # remove to demonstrate saving and loading

    model = SAC.load("./model/sb3_pendulum_sac")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == '__main__':
    # cartpole_ppo()
    # cartpole_dqn()
    cartpole_a2c()
    # cartpole_sac()
    # pendulum_sac()
