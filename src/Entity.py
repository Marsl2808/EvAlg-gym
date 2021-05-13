from src.NN import Neural_Network


class Entity(object):

    def __init__(self, n_layer_nodes, weight_interval):
        self.controller = Neural_Network(n_layer_nodes, weight_interval)
        self.fitness = -1000
        self.survived = False
        self.fitness_ecl_vec = 0.0
        self.hamming_dist = 0
