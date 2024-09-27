import gym
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from linear_rl.fourier import FourierBasis
import matplotlib.pyplot as plt
import numpy as np

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def main():

    env = gym.make('MountainCar-v0')
    agent = TrueOnlineSarsaLambda(env.observation_space, env.action_space,
                                alpha=0.001,
                                fourier_order=5,
                                gamma=0.99,
                                lamb=0.9,
                                epsilon=0.0,
                                min_max_norm=True)

    obs = env.reset()
    ret = 0
    rets = []
    episodes = 5000
    ep =  0
    while ep < episodes:
        action = agent.act(obs)
        new_obs, rew, done, info = env.step(action)
        ret += rew

        agent.learn(obs, action, rew, new_obs, done)

        obs = new_obs
        if done:
            print("Return:", ret)
            rets.append(ret)
            ret = 0
            ep += 1
            obs = env.reset()
        #env.render('human')
    
    plt.figure()
    plt.plot(moving_average(rets, 1))
    plt.ylabel("Return")
    plt.xlabel("Episode")

    obs, done = env.reset(), False
    ep = 0
    x, y, z = [], [], []
    while ep != 100:
        action = agent.act(obs)
        new_obs, rew, done, info = env.step(agent.act(obs))
        x.append(obs[0])
        y.append(obs[1])
        z.append(-agent.get_q_value(agent.get_features(obs), action))
        if done:
            ep +=1 
            obs = env.reset()
        else:
            obs = new_obs
    
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # Plot the surface.
    surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':
    main()
