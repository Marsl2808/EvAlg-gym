import random
import logging
from copy import deepcopy


class Selection(object):

    def __init__(self):
        pass

    @staticmethod
    def rank_based_selection(population_in):
        # survival of the fittest
        population_in.sort(key=lambda x: x.fitness, reverse=True)
        logging.info(f"max fitness: {population_in[0].fitness}")

        new_population = []
        for i, individuum in enumerate(population_in):
            if random.random() < ((len(population_in) - i) /
                                  (len(population_in))):
                new_population.append(deepcopy(individuum))

        return new_population
