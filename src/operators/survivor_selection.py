import random
import logging
from copy import deepcopy


class Survivor_Selection(object):

    def __init__(self, selection_case):
        if selection_case == "NAIVE_RANK_BASED":
            self.selection = lambda x: self.naive_rank_based(x)

    def naive_rank_based(self, population_in):
        population_in.sort(key=lambda x: x.fitness, reverse=False)
        logging.info(f"max fitness: {population_in[-1].fitness}")

        new_population = []
        for i, individuum in enumerate(population_in):
            # if random.random() < ((len(population_in) - i) /
            #                       (len(population_in))):
            if random.random() < (i / (len(population_in) - 1)):
                new_population.append(deepcopy(individuum))

        return new_population
