import logging
from copy import deepcopy
from src.individuum.entity import Entity
from src.operators.survivor_selection import Survivor_Selection
from src.operators.crossover import Crossover
from src.operators.mutation import Mutation
from src.operators.parent_selection import Parent_Selection


class Population_Manager(object):

    def __init__(self, Const):

        self.Const = Const
        self.population = self.breed_init_population()

        # TODO: outsource Params
        self.survivor_selection_operator = Survivor_Selection("NAIVE_RANK_BASED")
        self.parent_selection_operator = Parent_Selection("EXP_RANKED", Const["N_PARENTS"])
        self.mutation_operator = Mutation("FIRST_IMPL", Const['MUTATION_RATE'])
        self.crossover_operator = Crossover("CROSSOVER", Const['PROB_NODE_COPY'])

    def breed_init_population(self):
        return [Entity(self.Const['N_LAYER_NODES'],
                self.Const['WEIGHT_INIT_INTERVAL'])
                for i in range(self.Const['POP_SIZE'])]

    def breed_new_population(self):
        pop_size_in = len(self.population)
        self.population = self.survivor_selection_operator.selection(self.population)
        pop_tmp = deepcopy(self.population)
        logging.info(f"{len(self.population)} survived walker")

        while(len(self.population) < pop_size_in):
            parents = self.parent_selection_operator.selection(pop_tmp)
            self.population.append(self.crossover_operator.crossover(parents))

        for individuum in self.population:
            if not individuum.survived:
                self.mutation_operator.mutation(individuum.controller)
