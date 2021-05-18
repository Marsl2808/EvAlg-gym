import random


class Mutation(object):

    def __init__(self, mutation_case, mutation_rate):
        self.mutation_rate = mutation_rate
        if mutation_case == "MUTATION":
            self.mutation = lambda x: self.mutation_first_impl(x)

    def mutation_first_impl(self, nn):

        for layer_idx in range(len(nn.weights)):
            for node_idx in range(len(nn.weights[layer_idx])):
                if layer_idx != 1 and layer_idx < (nn.n_hidden - 1) and (
                  random.random() > self.mutation_rate):
                    self.mutate_bias(nn, layer_idx, node_idx)

            for weight_idx in range(len(nn.weights[layer_idx][node_idx])):
                if (random.random() > self.mutation_rate):
                    self.mutate_weight(nn, layer_idx, node_idx, weight_idx)

    def mutate_bias(self, nn, i, j):
        random_number = random.randint(1, 5)
        if random_number == 1:
            nn.bias[i][j] = random.random()
        elif random_number == 2:
            nn.bias[i][j] += random.random()
        elif random_number == 3:
            nn.bias[i][j] -= random.random()
        elif random_number == 4:
            nn.bias[i][j] = 0.0
        elif random_number == 5:
            nn.bias[i][j] *= (-1)
        # random.gauss(mu, sigma)

    def mutate_weight(self, nn, i, j, k):
        random_number = random.randint(1, 6)
        if random_number == 1:
            nn.weights[i][j][k] = random.random()
        elif random_number == 2:
            nn.weights[i][j][k] += random.random()
        elif random_number == 3:
            nn.weights[i][j][k] -= random.random()
        elif random_number == 4:
            nn.weights[i][j][k] = 0.0
        elif random_number == 5:
            nn.weights[i][j][k] *= (-1)
        elif random_number == 6:
            random_i = random.randint(0, len(nn.weights)-1)
            random_j = random.randint(0, len(nn.weights
                                      [random_i]) - 1)
            random_k = random.randint(0, len(nn.weights
                                      [random_i][random_j])-1)
            nn.weights[i][j][k] = (nn.weights[random_i][random_j][random_k])
