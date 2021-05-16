def adapt_obs(observation_env, entity):
    observation_norm = []
    for j in range(len(observation_env)):
        if entity.obs_norm[j].std != 0:
            observation_norm.append((observation_env[j]-entity.obs_norm[j].mean) / entity.obs_norm[j].std)    
        else:
            observation_norm.append(observation_env[j])

        entity.updated_obs_norm[j](observation_env[j])

    return observation_norm

def agent_env_loop(entity, generation):
    entity.survived = True
    entity.fitness = 0
    env.seed(10)
    observation_env = env.reset()  # s_0

    for i in range(MAX_SEQUENCE_LEN):

        if WELFORD:
            observation_norm = adapt_obs(observation_env, entity)   
        else:
            observation_norm = observation_env 
        
        entity.action_sequ.append(entity.controller.feed_forward(observation_norm)) # a_t, s_t 
        observation_env, reward, done, _ = env.step(entity.action_sequ[-1])                     # s_{t+1}, r_{t+1}            

        entity.fitness += reward
        if done:
            break


import gym
import time
from src.entity import Entity
from src.evolutionCtrl import Population_Manager
from src.trainEval import TrainEval
import logging
from src.optional.welford import Welford
#############
import copy
#############

logging.basicConfig(level=logging.INFO)
gym.logger.set_level(40)

# 1) Environment
env = gym.make("BipedalWalker-v3")
print(f"Dim. action space: {env.action_space.shape}")
print(f"Dim. observation space: {env.observation_space.shape}")

# 2) Training Params
POP_SIZE = 100
MAX_SEQUENCE_LEN = 400
N_GENERATIONS = 2500
MUTATION_RATE = .1
PROB_NODE_COPY = .1
WEIGHT_INIT_INTERVAL = [-1.0, 1.0]
WELFORD = False
print(f"Welford-normalizer set to : {WELFORD}")
N_LAYER_NODES = [env.observation_space.shape[0],
                 50,
                 env.action_space.shape[0]]
# 48, 48, 32,  |  70-80 fitness o. Welford  
#  32,32, 
# 3) Init population
init_population = [Entity(N_LAYER_NODES, WEIGHT_INIT_INTERVAL)
                   for i in range(POP_SIZE)]

pop_manager = Population_Manager(init_population, MUTATION_RATE,
                                 PROB_NODE_COPY)

train_evaluator = TrainEval()

# 4) Start training
if __name__ == '__main__':
    for generation in range(N_GENERATIONS):
        t_start = time.time()

        print(f"--- Generation {generation} ---")
        for entity in pop_manager.population:
            if not entity.survived:
                agent_env_loop(entity, generation)

        if generation % 50 == 0:
            train_evaluator.eval_training(pop_manager.population)

        pop_manager.breed_new_population()

        print(f"Time: {(time.time() - t_start):.2f} sec")