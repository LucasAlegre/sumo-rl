from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    @abstractmethod
    def new_episode(self):
        pass

    @abstractmethod
    def observe(self, observation):
        ''' To override '''
        pass

    @abstractmethod
    def act(self):
        ''' To override '''
        pass

    @abstractmethod
    def learn(self, action, reward, done):
        ''' To override '''
        pass
