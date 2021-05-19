import random


class Mutation(object):

    def __init__(self, mutation_case, mutation_rate):
        self.mutation_rate = mutation_rate
        if mutation_case == "FIRST_IMPL":
            self.mutation = lambda x: self.mutation_first_impl(x)
        if mutation_case == "NONUNIFORM":
            self.mutation = lambda x: self.nonuniform_mutation(x)

    def nonuniform_mutation(self, nn):

        # TODO
        mu = 0
        sigma = .3

        for i in range(len(nn.weights)):
            for j in range(len(nn.weights[i])):
                # if i < nn.n_hidden:
                #     nn.bias[i][j] += random.gauss(mu, sigma)

                nn.weights[i][j] = [weight + random.gauss(mu, sigma) for weight
                                    in nn.weights[i][j]]

    def mutation_first_impl(self, nn):
        for i in range(len(nn.weights)):
            nn.bias[i] = [self.mutate_bias(x)
                          if random.random() < self.mutation_rate else x
                          for x in nn.bias[i]]
            for j in range(len(nn.weights[i])):
                nn.weights[i][j] = [self.mutate_weight(x)
                                    if random.random() < self.mutation_rate
                                    else x for x in nn.weights[i][j]]

    def mutate_bias(self, bias):
        random_number = random.randint(1, 4)
        if random_number == 1:
            return random.random()
        elif random_number == 2:
            return bias + random.random() * 2 - 1
        elif random_number == 3:
            return 0.0
        elif random_number == 4:
            return -bias

    def mutate_weight(self, weight):
        random_number = random.randint(1, 4)
        if random_number == 1:
            return random.random()
        elif random_number == 2:
            return weight + random.random() * 2 - 1
        elif random_number == 3:
            return 0.0
        elif random_number == 4:
            return -weight
        # elif random_number == 5:
        #     random_i = random.randint(0, len(nn.weights)-1)
        #     random_j = random.randint(0, len(nn.weights
        #                               [random_i]) - 1)
        #     random_k = random.randint(0, len(nn.weights
        #                               [random_i][random_j])-1)
        #     return (nn.weights[random_i][random_j][random_k])
