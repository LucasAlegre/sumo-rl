import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

env_id = "CartPole-v1"
video_folder = "logs/videos/"
video_length = 100

vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

obs = vec_env.reset()

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env, video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=f"random-agent-{env_id}")



model = DQN("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=int(2e5), progress_bar=True)

# vec_env.reset()
# for _ in range(video_length + 1):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, _, _, _ = vec_env.step(action)
# # Save the video
# vec_env.close()

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(video_length + 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
