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
        
        # self.selection_fitness_prob = fitness_selection_prob
        # self.selection_diversity_prob = diversity_selection_prob
        
        # self.mutation_weight_interval = [-2.0, 2.0]
        # self.uper_bound_weight_interval = 1
        

    def breed_new_population(self):        
        
        new_population = self.selection()       
     
        # 2) crossover & mutation
        while(len(new_population) < self.pop_size):
            child = self.crossover(random.choice(new_population), random.choice(new_population))
            new_population.append(child)

        self.population = new_population      


    def selection(self):
        # survival of the fittest
        self.population.sort(key=lambda x: x.fitness, reverse=True)   
        print(f"max fitness is: {self.population[0].fitness}") 

        # save best 2 individuums
        new_population = [x for x in self.population[0:1]]      
        new_population = []       
        for i in range(2, len(self.population)):            
            if random.random() < ((self.pop_size - i) / (self.pop_size)):                                       
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
                #2) #loop over weights of node and choose random parent (50% probability each)
                else:
                    for k in range(len(child.controller.weights[i][j])): #loop over weights of node                        
                        random_parent = random.choice([parent_1, parent_2])    
                        child.controller.weights[i][j][k] = random_parent.controller.weights[i][j][k]
        return child