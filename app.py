import gym
import time
from src.Entity import Entity
from src.EvolutionCtrl import Population_Manager
from src.TrainEval import TrainEval
import logging

logging.basicConfig(level=logging.INFO)

# 1) Environment
env = gym.make("BipedalWalker-v3")
print(f"Dim. action space: {env.action_space.shape}")
print(f"Dim. observation space: {env.observation_space.shape}")

# 2) Training Params
POP_SIZE = 100
MAX_SEQUENCE_LEN = 400
N_GENERATIONS = 1500
MUTATION_RATE = .1
PROB_NODE_COPY = .1
WEIGHT_INIT_INTERVAL = [-1, 1]

n_layer_nodes = [env.observation_space.shape[0],
                 48, 48, 32,
                 env.action_space.shape[0]]

init_population = [Entity(n_layer_nodes, WEIGHT_INIT_INTERVAL)
                   for i in range(POP_SIZE)]

pop_manager = Population_Manager(init_population, MUTATION_RATE,
                                 PROB_NODE_COPY)

train_evaluator = TrainEval()


def agent_env_loop(entity):
    entity.survived = True
    entity.fitness = 0
    env.seed(10)
    observation = env.reset()  # s_0
    for i in range(MAX_SEQUENCE_LEN):
        action = entity.controller.feed_forward(observation)  # a_t, s_t
        observation, reward, done, _ = env.step(action)  # s_{t+1}, r_{t+1}
        entity.fitness += reward
        if done:
            break


if __name__ == '__main__':
    # Optimize
    for generation in range(N_GENERATIONS):
        t_start = time.time()
        print(f"--- Generation {generation} ---")
        for entity in pop_manager.population:
            if not entity.survived:
                agent_env_loop(entity)

        # if generation % 50 == 0:
        #     train_evaluator.eval_training(pop_manager.population, generation)

        pop_manager.breed_new_population(generation)

        print(f"Time: {time.time() - t_start}")
