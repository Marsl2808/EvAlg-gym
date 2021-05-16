from src.neuralNet import Neural_Network
from src.optional.welford import Welford


class Entity(object):

    def __init__(self, n_layer_nodes, weight_interval):
        self.controller = Neural_Network(n_layer_nodes, weight_interval)
        self.fitness = -1000
        self.survived = False

        # diversity
        self.hamming_dist = 0
        self.action_sequ = []

        self.obs_norm = [Welford() for i in range(n_layer_nodes[0])]
        self.updated_obs_norm = [Welford() for i in range(n_layer_nodes[0])]
