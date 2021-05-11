from NN import Neural_Network

class Entity(object):

    def __init__(self, n_layer_nodes, weight_interval):
        
        self.controller = Neural_Network(n_layer_nodes, weight_interval)
        self.fitness = -1000
        
        self.survived = False
        
#        self.length_of_fitness_ecl_vector = 0.0
#        self.diversity_of_vector_difference = 0.0  
#        self.euclidean_dist = 0
#        self.survival_prob = 0.0 # sum of fitness_prob and diversity_prob

#        self.obs_normalizer = [Welford() for i in range(input_nodes)]
#        self.updated_obs_normalizer = [Welford() for i in range(input_nodes)]
               
#        self.action_batch = []   #for diversity calc.