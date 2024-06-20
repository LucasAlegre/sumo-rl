import gym
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

"""
这段程序是xueshufan.com网站的Chat科研工助手根据我的一句话提示写的代码。
提示语是：用python写一段使用PPO算法的CartPole-v1的训练程序
在本工程环境下，该程序不能运行。
"""
# 定义 Policy 网络
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


# 定义 PPO 算法
class PPO:
    def __init__(self, env_name, policy_net, value_net, lr_actor, lr_critic, gamma, epsilon, k_epochs):
        self.env = gym.make(env_name)
        self.policy_net = policy_net
        self.value_net = value_net
        self.optimizer_actor = Adam(self.policy_net.parameters(), lr=lr_actor)
        self.optimizer_critic = Adam(self.value_net.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_epochs = k_epochs

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        G = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        return returns

    def train(self):
        state = self.env.reset()
        score = 0
        for epoch in range(self.k_epochs):
            log_probs = []
            values = []
            rewards = []
            dones = []
            for _ in range(self.env.spec.max_episode_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                log_prob = torch.log(self.policy_net(torch.FloatTensor(state)).squeeze(0))[action]
                value = self.value_net(torch.FloatTensor(state)).squeeze(0)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(done)

                state = next_state
                score += reward

                if done:
                    break

            next_value = self.value_net(torch.FloatTensor(next_state)).squeeze(0)
            returns = self.compute_returns(rewards, dones, next_value)

            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            returns = torch.FloatTensor(returns).unsqueeze(1)

            advantage = returns - values

            actor_loss = -torch.min(torch.exp(log_probs) * advantage, torch.clamp(torch.exp(log_probs) * advantage, 1 - self.epsilon, 1 + self.epsilon)).mean()
            critic_loss = advantage.pow(2).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        return score


# 环境名称
env_name = 'CartPole-v1'
# 输入维度和动作维度
input_dim = 4
output_dim = 2
# 初始化 Policy 网络和 Value 网络
policy_net = Policy(input_dim, output_dim)
value_net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))
# 学习率
lr_actor = 0.0003
lr_critic = 0.001
# 折扣因子
gamma = 0.99
# PPO 参数
epsilon = 0.2
k_epochs = 10
# 训练次数
num_episodes = 1000

ppo = PPO(env_name, policy_net, value_net, lr_actor, lr_critic, gamma, epsilon, k_epochs)

for episode in range(num_episodes):
    score = ppo.train()
    print(f"Episode {episode + 1}: score = {score}")
