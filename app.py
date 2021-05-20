import gym
import time
from src.evolutionCtrl import Population_Manager
from src.trainEval import TrainEval
import logging

logging.basicConfig(level=logging.INFO)
gym.logger.set_level(40)

# create gym Environment
env = gym.make("BipedalWalker-v3")
print(f"Dim. action space: {env.action_space.shape}")
print(f"Dim. observation space: {env.observation_space.shape}")

# training Params
Const = {
    'POP_SIZE': 10,
    'MUTATION_RATE': .1,
    'PROB_NODE_COPY': .1,
    'N_PARENTS': 3,
    'WEIGHT_INIT_INTERVAL': [-1.0, 1.0],
    'N_LAYER_NODES': [env.observation_space.shape[0],
                      50,
                      env.action_space.shape[0]]
}

MAX_SEQUENCE_LEN = 400
N_GENERATIONS = 5000

pop_manager = Population_Manager(Const)

train_evaluator = TrainEval()


def agent_env_loop(entity, generation):
    entity.survived = True
    entity.fitness = 0
    env.seed(10)
    observation = env.reset()                                 # s_0

    for i in range(MAX_SEQUENCE_LEN):

        action = entity.controller.feed_forward(observation)  # a_t, s_t
        observation, reward, done, _ = env.step(action)  # s_{t+1}, r_{t+1}

        entity.fitness += reward
        if done:
            break


if __name__ == '__main__':
    for generation in range(N_GENERATIONS):
        t_start = time.time()

        print(f"--- Generation {generation} ---")
        for entity in pop_manager.population:
            if not entity.survived:
                agent_env_loop(entity, generation)

        if generation % 100 == 0:
            train_evaluator.eval_training(pop_manager.population)

        pop_manager.breed_new_population()

        print(f"Time: {(time.time() - t_start):.2f} sec")
