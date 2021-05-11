import random
from Entity import Entity
import copy

class Population_Manager(object) :

    def __init__(self, initial_population,
                 mutation_rate, prob_node_copy, 
                 fitness_selection_prob, diversity_selection_prob): 

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
                #1) inherit complete node with defined probability (currently no mutation)
                if random.random() > self.prob_node_copy:
                    random_parent = random.choice([parent_1, parent_2])
                    child.controller.weights[i][j] = copy.deepcopy(random_parent.controller.weights[i][j])
                # loop over weights 
                else:
                    for k in range(len(child.controller.weights[i][j])): #loop over weights of node       
                        if random.random() > self.mutation_rate: # crossover                   
                             random_parent = random.choice([parent_1, parent_2]) 
                             child.controller.weights[i][j][k] = random_parent.controller.weights[i][j][k]
                        else:
                            random_parent_weight = random.choice([parent_1.controller.weights[i][j][k], parent_2.controller.weights[i][j][k]])
                            child.controller.weights[i][j][k] = self.mutate_weight(child.controller.weights[i][j][k], random_parent_weight)
        return child
    

    def mutate_weight(self, weight, random_weight):                   
        random_number = random.randint(1,5)
        # random weights from initialization
        if random_number == 1:    
            return weight
        # add +/- random_nr[0,1] to parent_1 weights
        elif random_number == 2: 
            weight = random_weight + random.random()
        elif random_number == 3:
            weight = random_weight - random.random()
        # deactivate weight
        elif random_number == 4:                                
            weight = 0.0
        # change sign
        elif random_number == 5:                               
            weight = random_weight *(-1)                            
        return weight 

        # # change random node of const. layer
        # elif random_number == 9:
        #     random_j = random.randint(0,len(child.weights[i])-1)
        #     random_k = random.randint(0,len(child.weights[i][j])-1)
        #     child.weights[i][j][k] = parent_1.weights[i][random_j][random_k]
        # elif random_number == 10:
        #     random_j = random.randint(0,len(child.weights[i])-1)
        #     random_k = random.randint(0,len(child.weights[i][j])-1)
        #     child.weights[i][j][k] = parent_2.weights[i][random_j][random_k] 
           