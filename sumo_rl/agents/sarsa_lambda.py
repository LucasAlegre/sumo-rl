import numpy as np
from itertools import combinations, product

class TrueOnlineSarsaLambda:

    def __init__(self, state_space, action_space, alpha=0.001, lamb=0.9, gamma=0.99, epsilon=0.05, fourier_order=7):
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = epsilon

        self.state_space = state_space
        self.state_dim = self.state_space.shape[0]
        self.action_space = action_space
        self.fourier_order = fourier_order

        self.coeffs = self._build_coefficients()
        self.lr = self._build_learning_rates()
    
        self.et = {a: np.zeros(len(self.coeffs)) for a in range(self.action_space.n)}
        self.theta = {a: np.zeros(len(self.coeffs)) for a in range(self.action_space.n)}

        self.q_old = None
        self.action = None

    def _build_coefficients(self):
        coeff = np.array(np.zeros(self.state_dim))  # Bias
        for i in range(1, 2 + 1):
            for indices in combinations(range(self.state_dim), i):
                for c in product(range(1, self.fourier_order + 1), repeat=i):
                    coef = np.zeros(self.state_dim)
                    coef[list(indices)] = list(c)
                    coeff = np.vstack((coeff, coef))
        return coeff

    def _build_learning_rates(self):
        lrs = np.linalg.norm(self.coeffs, axis=1)
        lrs[lrs==0.] = 1.
        lrs = self.alpha/lrs
        return lrs

    def learn(self, state, action, reward, next_state, done):
        phi = self.get_features(state)
        next_phi = self.get_features(next_state)
        q = self.get_q_value(phi, action)
        next_q = self.get_q_value(next_phi, self.act(next_phi))
        td_error = reward + self.gamma * next_q - q
        if self.q_old is None:
            self.q_old = q

        for a in range(self.action_space.n):
            if a == action:
                self.et[a] = self.lamb*self.gamma*self.et[a] + phi - self.lr*self.gamma*self.lamb*(self.et[a]*phi)*phi
                self.theta[a] += self.lr*(td_error + q - self.q_old)*self.et[a] - self.lr*(q - self.q_old)*phi
            else:
                self.et[a] = self.lamb*self.gamma*self.et[a]
                self.theta[a] += self.lr*(td_error + q - self.q_old)*self.et[a]
        
        self.q_old = next_q
        if done:
            self.reset_traces()

    def get_q_value(self, features, action):
        return np.dot(self.theta[action], features)
        
    def get_features(self, state):
        return np.cos(np.dot(np.pi*self.coeffs, state))
    
    def reset_traces(self):
        self.q_old = None
        self.et = {a: np.zeros(len(self.coeffs)) for a in range(self.action_space.n)}
    
    def act(self, features):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = [self.get_q_value(features, a) for a in range(self.action_space.n)]
            return q_values.index(max(q_values))
