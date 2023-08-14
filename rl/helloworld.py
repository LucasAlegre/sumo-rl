import gym

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")          # 导入环境

episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    truncated = False
    score = 0

    while not done | truncated:
        env.render()                           # 渲染环境
        action = env.action_space.sample()     # 随机采样动作
        n_state, reward, done, truncated, info = env.step(action)    # 和环境交互，得到下一个状态，奖励等信息
        score += reward                        # 计算分数
    print("Episode : {}, Score : {}".format(episode, score))

env.close()     # 关闭窗口
