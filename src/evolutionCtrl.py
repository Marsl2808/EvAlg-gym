import random
import logging
from copy import deepcopy
from src.individuum.entity import Entity
from src.operators.selection import Selection
from src.operators.crossover import Crossover
from src.operators.mutation import Mutation


class Population_Manager(object):

    def __init__(self, Const):

        self.Const = Const
        self.population = self.breed_init_population()

        # TODO: outsource Params
        # evolutionary operators
        self.selection_obj = Selection("RANK_BASED_SELECTION")
        self.mutation_obj = Mutation("FIRST_IMPL", Const['MUTATION_RATE'])
        self.crossover_obj = Crossover("CROSSOVER", Const['PROB_NODE_COPY'])

    def breed_init_population(self):
        return [Entity(self.Const['N_LAYER_NODES'],
                self.Const['WEIGHT_INIT_INTERVAL'])
                for i in range(self.Const['POP_SIZE'])]

    def breed_new_population(self):
        pop_size_in = len(self.population)
        self.population = self.selection_obj.selection(self.population)
        logging.info(f"{len(self.population)} survived walker")

        while(len(self.population) < pop_size_in):
            parents = [deepcopy(random.choice(self.population))
                       for i in range(self.Const['N_PARENTS'])]
            self.population.append(self.crossover_obj.crossover(parents))

        for individuum in self.population:
            # ???
            if not individuum.survived:
                self.mutation_obj.mutation(individuum.controller)
