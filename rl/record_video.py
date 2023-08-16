import gymnasium as gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

env_id = "CartPole-v1"
video_folder = "./videos/"
video_length = 10000

vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

obs = vec_env.reset()

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env, video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=f"random-agent-{env_id}")

vec_env.reset()
for _ in range(video_length + 1):
    action = [vec_env.action_space.sample()]
    obs, _, _, _ = vec_env.step(action)
# Save the video
vec_env.close()
