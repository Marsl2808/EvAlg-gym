import random
import logging
from copy import deepcopy


class Selection(object):

    def __init__(self, selection_case):
        if selection_case == "RANK_BASED_SELECTION":
            self.selection = lambda x: self.rank_based_selection(x)

    def rank_based_selection(self, population_in):
        population_in.sort(key=lambda x: x.fitness, reverse=True)
        logging.info(f"max fitness: {population_in[0].fitness}")

        new_population = []
        for i, individuum in enumerate(population_in):
            if random.random() < ((len(population_in) - i) /
                                  (len(population_in))):
                new_population.append(deepcopy(individuum))

        return new_population
