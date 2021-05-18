import numpy as np
import copy


class Neural_Network(object):

    def __init__(self, n_layer_nodes, weight_interval):
        self.n_hidden = len(n_layer_nodes) - 2
        self.n_layer_nodes = n_layer_nodes
        self.weight_interval = weight_interval

        self.weights = []
        self.bias = []
        for i in range(len(n_layer_nodes) - 1):
            self.weights.append(np.random.uniform(weight_interval[0],
                                                  weight_interval[1],
                                                  (n_layer_nodes[i],
                                                  n_layer_nodes[i+1])))
            self.bias.append(np.random.uniform(weight_interval[0],
                                               weight_interval[1],
                                               n_layer_nodes[i+1]))

        # activation-functions
        self.sigmoid = lambda x: (1/(1 + np.exp(-x)))
        self.tanh = lambda x: np.tanh(x)
        self.relu = lambda x: np.maximum(x, 0)

    def feed_forward(self, observation):
        layer_out = observation

        # TODO: layer specific activation functions
        for i in range(len(self.weights)-1):
            layer_in = np.dot(layer_out, self.weights[i]) + self.bias[i]
            layer_out = self.relu(layer_in)

        final_in = np.dot(layer_out, self.weights[i+1]) + self.bias[i+1]
        final_out = self.tanh(final_in)

        return final_out

    def set_weights(self, weights):
        self.weights = copy.deepcopy(weights)
