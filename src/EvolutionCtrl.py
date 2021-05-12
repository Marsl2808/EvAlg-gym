import random
from copy import deepcopy
from src.Entity import Entity


class Population_Manager(object):

    def __init__(self, initial_population, mutation_rate, prob_node_copy):
        self.population = initial_population
        self.pop_size = len(initial_population)
        self.prob_node_copy = prob_node_copy
        self.mutation_rate = mutation_rate

    def breed_new_population(self, generation_act):
        pop_size_in = len(self.population)
        self.population = self.selection()

        while(len(self.population) < pop_size_in):
            child = self.crossover(random.choice(self.population),
                                   random.choice(self.population),
                                   generation_act)
            self.population.append(child)

    def selection(self):
        # survival of the fittest
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = []

        for i in range(len(self.population)):
            if random.random() < ((len(self.population) - i) /
                                  (len(self.population))):
                new_population.append(self.population[i])

        print(f"{len(new_population)} survived walker," +
              f"max fitness: {new_population[0].fitness}")

        return new_population

    def crossover(self, parent_1, parent_2, gen_act):
        child = Entity(parent_1.controller.n_layer_nodes,
                       parent_1.controller.weight_interval)

        for l in range(len(child.controller.weights)):
            if l == child.controller.n_hidden:
                self.set_bias_output(parent_1, parent_2, child, l)

            for n in range(len(child.controller.weights[l])):
                random_parent = random.choice([parent_1, parent_2])

                if l != 1 and l < child.controller.n_hidden - 1:
                    self.set_bias_hidden(child, l, n, random_parent)

                if random.random() > self.prob_node_copy:
                    self.copy_node(gen_act, child, l, n, random_parent)

                else:
                    for w in range(len(child.controller.weights[l][n])):
                        random_parent = random.choice([parent_1, parent_2])
                        if random.random() > self.mutation_rate:
                            self.set_weights(random_parent, l, n, w, child)
                        else:
                            self.mutate_weight(child, random_parent, l, n, w)
        return child

    def set_weights(self, random_parent, i, j, k, child):
        child.controller.weights[i][j][k] = (random_parent.controller.weights
                                             [i][j][k])

    def set_bias_output(self, parent_1, parent_2, child, i):
        for j in range(len(child.controller.bias[i])):
            random_parent = random.choice([parent_1, parent_2])
            child.controller.bias[i][j] = random_parent.controller.bias[i][j]

    def set_bias_hidden(self, child, i, j, random_parent):
        child.controller.bias[i][j] = (random_parent.controller.bias[i][j])
        if (random.random() > self.mutation_rate):
            self.mutate_bias(child, random_parent, i, j)

    def copy_node(self, gen_act, child, i, j, random_parent):
        child.controller.weights[i][j] = deepcopy(random_parent.controller
                                                  .weights[i][j])
        # first layer no bias, last layer differnet size TODO
        if i != 1 and i < child.controller.n_hidden - 1:
            child.controller.bias[i][j] = random_parent.controller.bias[i][j]

    def mutate_weight(self, child, parent, i, j, k):
        random_number = random.randint(1, 6)
        parent_weight = parent.controller.weights[i][j][k]
        # random weights from initialization
        if random_number == 1:
            return
        # add +/- random_nr[0,1] to parent_1 weights
        elif random_number == 2:
            child.controller.weights[i][j][k] = parent_weight + random.random()
        elif random_number == 3:
            child.controller.weights[i][j][k] = parent_weight - random.random()
        # deactivate weight
        elif random_number == 4:
            child.controller.weights[i][j][k] = 0.0
        # change sign
        elif random_number == 5:
            child.controller.weights[i][j][k] = parent_weight * (-1)
        # change random weight
        elif random_number == 6:
            random_i = random.randint(0, len(child.controller.weights)-1)
            random_j = random.randint(0, len(child.controller.weights[random_i]
                                             ) - 1)
            random_k = random.randint(0, len(child.controller.weights[random_i]
                                             [random_j])-1)
            child.controller.weights[i][j][k] = (parent.controller.weights
                                                 [random_i][random_j][random_k]
                                                 )

    def mutate_bias(self, child, parent, i, j):
        case_of_mutation = random.randint(1, 5)
        parent_bias = parent.controller.bias[i][j]
        # random bias
        if case_of_mutation == 1:
            return
        # add +/- random_nr[0,1] to parent_1 bias
        elif case_of_mutation == 2:
            child.controller.bias[i][j] = parent_bias + random.random()
        elif case_of_mutation == 3:
            child.controller.bias[i][j] = parent_bias - random.random()
        # deactivate bias
        elif case_of_mutation == 4:
            child.controller.bias[i][j] = 0.0
        # change sign of parent bias
        elif case_of_mutation == 5:
            child.controller.bias[i][j] = parent_bias * (-1)
