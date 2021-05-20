
import random
from copy import deepcopy


class Parent_Selection():

    def __init__(self, selection_case, n_parents):
        self.n_parents = n_parents
        if selection_case == "LINEAR_RANKED":
            self.selection = lambda x: self.linear_ranked(x)

    def linear_ranked(self, population):
        mu = len(population)
        s = 2
        probs = [((2 - s) / mu) + (2 * i * (s - 1))/(mu*(mu - 1))
                 for i in range(mu)]

        return deepcopy(random.choices(population, weights=probs,
                                       cum_weights=None, k=self.n_parents))
