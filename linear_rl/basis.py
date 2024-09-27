class Basis:

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.state_dim = state_space.shape[0]
        self.action_dim = self.action_space.n
    
    def get_features(self, state):
        pass

    def get_num_basis(self) -> int:
        pass
