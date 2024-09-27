import numpy as np
from itertools import combinations, product

from linear_rl.basis import Basis


class FourierBasis(Basis):

    def __init__(self, state_space, action_space, order, max_non_zero=2):
        super().__init__(state_space, action_space)
        self.order = order
        self.max_non_zero = min(max_non_zero, state_space.shape[0])
        self.coeff = self._build_coefficients()
    
    def get_learning_rates(self, alpha):
        lrs = np.linalg.norm(self.coeff, axis=1)
        lrs[lrs==0.] = 1.
        lrs = alpha/lrs
        return lrs
    
    def _build_coefficients(self):
        coeff = np.array(np.zeros(self.state_dim))  # Bias

        for i in range(1, self.max_non_zero + 1):
            for indices in combinations(range(self.state_dim), i):
                for c in product(range(1, self.order + 1), repeat=i):
                    coef = np.zeros(self.state_dim)
                    coef[list(indices)] = list(c)
                    coeff = np.vstack((coeff, coef))
        return coeff
    
    def get_features(self, state):
        return np.cos(np.dot(np.pi*self.coeff, state))

    def get_num_basis(self) -> int:
        return len(self.coeff)
