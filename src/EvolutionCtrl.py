import random
from Entity import Entity
import copy

class Population_Manager(object) :

    def __init__(self, initial_population, mutation_rate, prob_node_copy): 
        self.population = initial_population
        self.pop_size = len(initial_population)    
        self.prob_node_copy = prob_node_copy    
        self.mutation_rate = mutation_rate


    def breed_new_population(self):       
        pop_size_in = len(self.population)
        self.population = self.selection()       

        while(len(self.population) < pop_size_in):
            child = self.crossover(random.choice(self.population), random.choice(self.population))
            self.population.append(child)    


    def selection(self):      
        # survival of the fittest
        self.population.sort(key=lambda x: x.fitness, reverse=True)    
        new_population = []

        for i in range(len(self.population)):            
            if random.random() < ((len(self.population) - i) / (len(self.population))):                                       
                new_population.append(self.population[i])
 
        print(f"{len(new_population)} survived walker, max fitness is: {new_population[0].fitness}") 

        return new_population

    def crossover(self, parent_1, parent_2):
        child = Entity(parent_1.controller.n_layer_nodes, parent_1.controller.weight_interval)
        # loop over layers
        for i in range(len(child.controller.weights)):   
            # loop over nodes
            for j in range(len(child.controller.weights[i])): 

                # inherit complete node 
                if random.random() > self.prob_node_copy:
                    random_parent = random.choice([parent_1, parent_2])
                    child.controller.weights[i][j] = copy.deepcopy(random_parent.controller.weights[i][j])
                    
                    # first layer has no bias, last layer has differnet size TODO 
                    if i != 1 and i < child.controller.n_hidden - 1: 
                        child.controller.bias[i][j] = copy.deepcopy(random_parent.controller.bias[i][j])

                if (i != 1) and (i < child.controller.n_hidden - 1) and (random.random() > self.mutation_rate): 
                    random_parent = random.choice([parent_1, parent_2])
                    self.mutate_bias(child, random_parent, i, j)

                # loop over weights 
                else:
                    for k in range(len(child.controller.weights[i][j])):     
                        random_parent = random.choice([parent_1, parent_2]) 
                        if random.random() > self.mutation_rate:                 
                             child.controller.weights[i][j][k] = copy.deepcopy(random_parent.controller.weights[i][j][k])
                        else:
                            self.mutate_weight(child, random_parent, i, j, k)
        return child
    

    def mutate_weight(self, child, parent, i, j, k):       
        random_number = random.randint(1,6)
        parent_weight = copy.deepcopy(parent.controller.weights[i][j][k])
        # random weights from initialization
        if random_number == 1:    
            return
        # add +/- random_nr[0,1] to parent_1 weights
        elif random_number == 2: 
            child.controller.weights[i][j][k] = parent_weight + random.random()
        elif random_number == 3:
            child.controller.weights[i][j][k] = parent_weight - random.random()
        # deactivate weight
        elif random_number == 4:                                
            child.controller.weights[i][j][k] = 0.0
        # change sign
        elif random_number == 5:                               
            child.controller.weights[i][j][k] = parent_weight *(-1)                            
        # change random weight 
        elif random_number == 6:
            random_i = random.randint(0,len(child.controller.weights)-1)
            random_j = random.randint(0,len(child.controller.weights[random_i])-1)
            random_k = random.randint(0,len(child.controller.weights[random_i][random_j])-1)
            child.controller.weights[i][j][k] = copy.deepcopy(parent.controller.weights[random_i][random_j][random_k])


    def mutate_bias(self, child, parent, i, j): 
        case_of_mutation = random.randint(1,5)
        parent_bias = copy.deepcopy(parent.controller.bias[i][j])
        # random bias
        if case_of_mutation == 1:    
            return
        # add +/- random_nr[0,1] to parent_1 bias
        elif case_of_mutation == 2: 
            child.controller.bias[i][j] = parent_bias + random.random()
        elif case_of_mutation == 3:
            child.controller.bias[i][j] = parent_bias - random.random()
        # deactivate bias
        elif case_of_mutation == 4:                                
            child.controller.bias[i][j] = 0.0
        # change sign of parent bias
        elif case_of_mutation == 5:                               
            child.controller.bias[i][j] = parent_bias * (-1)                                



# TODO's: check copy, bias
