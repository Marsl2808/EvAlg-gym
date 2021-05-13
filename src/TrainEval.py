import matplotlib.pyplot as plt
from difflib import SequenceMatcher as SequMatch
import logging
import time


class TrainEval(object):

    def eval_training(self, population, generation):
        self.calc_diversity(population)
        self.plot_fitness_diversity(population, generation)

    def calc_diversity(self, population):
        start_t = time.time()
        for entity in population:
            weights_e1 = entity.controller.weights
            for entity_2 in population:
                entity_2.hamming_dist = 0
                weights_e2 = entity_2.controller.weights
                if id(entity) == id(entity_2):
                    break
                else:
                    # loop over layers
                    for l in range(len(weights_e1)):
                        # loop over nodes
                        for n in range(len(weights_e1[l])):
                            # TODO: inplace?
                            sm = SequMatch(None, weights_e1[l][n].round(3),
                                           weights_e2[l][n].round(3))
                            # (=2M/T) (T:# elems both sequences, M: # matches)
                            entity_2.hamming_dist += sm.ratio()

        logging.info(f"Time for diversity calculation: {time.time() - start_t}")

    def plot_fitness_diversity(self, population, generation):

        plt.plot([x.fitness for x in population],
                 [x.hamming_dist for x in population], 'ro')
        plt.title('Generation:' + str(generation))
        plt.grid(which='both')
        plt.xlabel('Fitness')
        plt.ylabel('Diversity')

        plt.show()
        plt.clf()
