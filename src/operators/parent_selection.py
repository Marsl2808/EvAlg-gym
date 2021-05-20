
import random
from copy import deepcopy
import numpy as np


class Parent_Selection():

    def __init__(self, selection_case, n_parents):
        self.n_parents = n_parents
        if selection_case == "LINEAR_RANKED":
            self.selection = lambda x: self.lin_rank(x)
        elif selection_case == "EXP_RANKED":
            self.selection = lambda x: self.exp_rank(x)

    def lin_rank(self, population):
        mu = len(population)
        s = 2
        probs = [((2 - s) / mu) + (2 * i * (s - 1))/(mu*(mu - 1))
                 for i in range(mu)]

        return deepcopy(random.choices(population, weights=probs,
                                       cum_weights=None, k=self.n_parents))

    def exp_rank(self, population):
        probs = [1 - np.exp(-i) for i in range(len(population))]
        probs_sum = sum(probs)
        probs_norm = [x / probs_sum for x in probs]

        return deepcopy(random.choices(population, weights=probs_norm,
                                       cum_weights=None, k=self.n_parents))
