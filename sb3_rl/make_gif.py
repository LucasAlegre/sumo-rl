import imageio
import numpy as np

from stable_baselines3 import A2C

model = A2C("MlpPolicy", "LunarLander-v2").learn(100_000)

images = []
obs = model.env.reset()
img = model.env.render(mode="rgb_array")
for i in range(3500):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _, _ = model.env.step(action)
    img = model.env.render(mode="rgb_array")

imageio.mimsave("./videos/lander_a2c.gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], duration=29)
