import random
from copy import deepcopy
from src.individuum.entity import Entity


class Crossover():

    def __init__(self):
        pass

    @staticmethod
    def crossover(parents):
        PROB_NODE_COPY = .1

        child = Entity(parents[0].controller.n_layer_nodes,
                       parents[0].controller.weight_interval)

        for layer_idx in range(len(child.controller.weights)):
            if layer_idx == child.controller.n_hidden:
                set_bias_output(parents, child, layer_idx)

            for node_idx in range(len(child.controller.weights[layer_idx])):
                random_parent = random.choice(parents)

                if layer_idx != 1 and layer_idx < child.controller.n_hidden - 1:
                    set_bias_hidden(child, layer_idx, node_idx, random_parent)

                if random.random() > PROB_NODE_COPY:
                    copy_node(child, layer_idx, node_idx, random_parent)

                else:
                    for weight_idx in range(len(child.controller.weights[layer_idx][node_idx])):
                        random_parent = random.choice(parents)
                        #if random.random() > self.mutation_rate:
                        set_weights(random_parent, layer_idx, node_idx, weight_idx, child)
                        #else:
                        #    self.mutate_weight(child, random_parent, l, n, w)
        return child


def set_bias_output(parents, child, i):
    for j in range(len(child.controller.bias[i])):
        random_parent = random.choice(parents)
        child.controller.bias[i][j] = random_parent.controller.bias[i][j]


def set_bias_hidden(child, i, j, random_parent):
    child.controller.bias[i][j] = (random_parent.controller.bias[i][j])
    # if (random.random() > self.mutation_rate):
    #     self.mutate_bias(child, random_parent, i, j)


def copy_node(child, i, j, random_parent):
    child.controller.weights[i][j] = deepcopy(random_parent.controller
                                              .weights[i][j])
    # first layer no bias, last layer differnet size TODO
    if i != 1 and i < child.controller.n_hidden - 1:
        child.controller.bias[i][j] = random_parent.controller.bias[i][j]


def set_weights(random_parent, i, j, k, child):
    child.controller.weights[i][j][k] = (random_parent.controller.weights
                                         [i][j][k])





    # def mutate_weight(self, child, parent, i, j, k):
    #     random_number = random.randint(1, 6)
    #     parent_weight = parent.controller.weights[i][j][k]
    #     # random weights (init)
    #     if random_number == 1:
    #         return
    #     elif random_number == 2:
    #         child.controller.weights[i][j][k] = parent_weight + random.random()
    #     elif random_number == 3:
    #         child.controller.weights[i][j][k] = parent_weight - random.random()
    #     elif random_number == 4:
    #         child.controller.weights[i][j][k] = 0.0
    #     elif random_number == 5:
    #         child.controller.weights[i][j][k] = parent_weight * (-1)
    #     elif random_number == 6:
    #         random_i = random.randint(0, len(child.controller.weights)-1)
    #         random_j = random.randint(0, len(child.controller.weights[random_i]
    #                                          ) - 1)
    #         random_k = random.randint(0, len(child.controller.weights[random_i]
    #                                          [random_j])-1)
    #         child.controller.weights[i][j][k] = (parent.controller.weights
    #                                              [random_i][random_j][random_k]
    #                                              )

    # def mutate_bias(self, child, parent, i, j):
    #     case_of_mutation = random.randint(1, 5)
    #     parent_bias = parent.controller.bias[i][j]
    #     # random bias (init)
    #     if case_of_mutation == 1:
    #         return
    #     elif case_of_mutation == 2:
    #         child.controller.bias[i][j] = parent_bias + random.random()
    #     elif case_of_mutation == 3:
    #         child.controller.bias[i][j] = parent_bias - random.random()
    #     elif case_of_mutation == 4:
    #         child.controller.bias[i][j] = 0.0
    #     elif case_of_mutation == 5:
    #         child.controller.bias[i][j] = parent_bias * (-1)
