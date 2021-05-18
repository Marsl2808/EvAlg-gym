import random
import logging
from src.selection import Selection
from src.crossover import Crossover


class Population_Manager(object):

    def __init__(self, initial_population, mutation_rate, prob_node_copy):
        self.population = initial_population
        self.prob_node_copy = prob_node_copy
        self.mutation_rate = mutation_rate

        # operators
        self.selection = lambda x: Selection.rank_based_selection(x)
        self.crossover = lambda x: Crossover.crossover(x)

    def breed_new_population(self):
        pop_size_in = len(self.population)
        self.population = self.selection(self.population)
        logging.info(f"{pop_size_in - len(self.population)} survived walker,")

        while(len(self.population) < pop_size_in):
            parents = [random.choice(self.population), random.choice(self.population)]
            child = self.crossover(parents)
            self.population.append(child)
