from agents.agent import Agent
from functools import reduce
import numpy as np

from exploration.epsilon_greedy import EpsilonGreedy

class QLAgent(Agent):

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        super(QLAgent, self).__init__(state_space, action_space)
        self.state = starting_state
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros([reduce(lambda x, y: x*y, [s.n for s in state_space.spaces]), action_space.n])
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def new_episode(self):
        pass

    def observe(self, observation):
        ''' To override '''
        pass

    def act(self):
        self.action = self.exploration.choose(self.q_table)
        return self.action

    def learn(self, new_state, reward, done=False):
        s = self.state
        s1 = new_state
        a = self.action
        self.q_table[s, a] = self.q_table[s, a] + self.alpha*(reward + self.gamma*np.max(self.q_table[s1, :]) - self.q_table[s, a])
        self.state = s1
        self.acc_reward += reward
