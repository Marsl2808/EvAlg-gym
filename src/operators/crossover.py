import random
from src.individuum.entity import Entity


class Crossover(object):

    def __init__(self, crossover_case, prob_node_copy):
        self.prob_node_copy = prob_node_copy

        if crossover_case == "CROSSOVER":
            self.crossover = lambda x: self.crossover_first_impl(x)

    def crossover_first_impl(self, parents):

        child = Entity(parents[0].controller.n_layer_nodes,
                       parents[0].controller.weight_interval)

        for layer_idx in range(len(child.controller.weights)):
            if layer_idx == child.controller.n_hidden:
                self.set_bias_output(parents, child, layer_idx)

            for node_idx in range(len(child.controller.weights[layer_idx])):
                random_parent = random.choice(parents)

                if layer_idx != 1 and layer_idx < (child.controller.n_hidden
                                                   - 1):
                    self.set_bias_hidden(child, layer_idx, node_idx,
                                         random_parent)

                if random.random() > self.prob_node_copy:
                    self.copy_node(child, layer_idx, node_idx, random_parent)

                else:
                    for weight_idx in range(len(child.controller.weights
                                            [layer_idx][node_idx])):
                        random_parent = random.choice(parents)
                        self.set_weights(random_parent, layer_idx, node_idx,
                                         weight_idx, child)
        return child

    def set_bias_output(self, parents, child, i):
        for j in range(len(child.controller.bias[i])):
            parent = random.choice(parents)
            child.controller.bias[i][j] = parent.controller.bias[i][j]

    def set_bias_hidden(self, child, i, j, parent):
        child.controller.bias[i][j] = (parent.controller.bias[i][j])

    def copy_node(self, child, i, j, parent):
        child.controller.weights[i][j] = parent.controller.weights[i][j]
        # first layer no bias, last layer differnet size TODO
        if i != 1 and i < child.controller.n_hidden - 1:
            child.controller.bias[i][j] = parent.controller.bias[i][j]

    def set_weights(self, parent, i, j, k, child):
        child.controller.weights[i][j][k] = (parent.controller.weights
                                             [i][j][k])
