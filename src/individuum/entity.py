from src.individuum.neuralNet import Neural_Network


class Entity(object):

    def __init__(self, n_layer_nodes, weight_interval):
        self.controller = Neural_Network(n_layer_nodes, weight_interval)
        self.fitness = -1000
        self.survived = False

        # diversity
        self.hamming_dist = 0
