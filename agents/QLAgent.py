from agents.agent import Agent
from functools import reduce


class QLAgent(Agent):

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, explorationStrategy=EpsilonGreedy()):
        super(QLAgent, self).__init__(state_space, action_space)
        self.state = starting_state
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {s: [0 for _ in range(action_space.n)] for s in range(reduce(lambda x, y: x*y, [s.n for s in state_space.spaces]))}

    def new_episode(self):
        pass

    def observe(self, observation):
        ''' To override '''
        pass

    def act(self):
        ''' To override '''
        pass

    def learn(self, action, reward, done):
        ''' To override '''
        pass
