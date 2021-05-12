import gym
import time
from Entity import Entity
from EvolutionCtrl import Population_Manager

# 1) Environment
env = gym.make("BipedalWalker-v3") 
env.seed(0)    # set seed for const. initial state

# 2) Training Params
POP_SIZE = 10#120
MAX_SEQUENCE_LEN = 400
N_GENERATIONS = 10#1500    

weight_interval = [-1, 1]                  # interval for initial and random weights
print(env.action_space.shape)
n_layer_nodes = [env.observation_space.shape[0], 48, 48, 32, env.action_space.shape[0]]
print(env.observation_space.shape)
init_population = [Entity(n_layer_nodes, weight_interval) for i in range(POP_SIZE)]  
pop_manager = Population_Manager(init_population, .1, .1)

def agent_env_loop(entity):
        entity.survived = True
        entity.fitness = 0
        env.seed(10) 
        observation = env.reset()                                            # s_0
        for i in range(MAX_SEQUENCE_LEN):  
            action = entity.controller.feed_forward(observation)             # a_t, s_t            
            observation, reward, done, _ = env.step(action)                  # s_{t+1}, r_{t+1}, a_t       
            entity.fitness += reward
            if done:
                break   

# Optimize 
for i in range(N_GENERATIONS):
    t_start = time.time() 
    for entity in pop_manager.population:
        if not entity.survived:
            agent_env_loop(entity)
    
    pop_manager.breed_new_population()
    print(f"Generation {i}: {time.time() - t_start}")
