from src.NN import Neural_Network
from src.optional.Welford import Welford


class Entity(object):

    def __init__(self, n_layer_nodes, weight_interval):
        self.controller = Neural_Network(n_layer_nodes, weight_interval)
        self.fitness = -1000
        self.survived = False
        self.hamming_dist = 0

        self.obs_norm = [Welford() for i in range(n_layer_nodes[0])]
        self.updated_obs_norm = [Welford() for i in range(n_layer_nodes[0])]
