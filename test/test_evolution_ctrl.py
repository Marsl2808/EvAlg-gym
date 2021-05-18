from src.evolutionCtrl import Population_Manager
from src.entity import Entity
import unittest


class Test_Population_Manager(unittest.TestCase):
    POP_SIZE = 10

    def test_ctor(self):
        pm = self.get_pop_manager()
        self.assertEqual(self.POP_SIZE, len(pm.population))

    def test_breed_new_population(self):
        pm = self.get_pop_manager()
        new_pop = pm.breed_new_population()
        self.assertEqual(self.POP_SIZE, len(new_pop.population))

    def get_pop_manager(self, pop_size):
        n_layer_nodes = [5, 5, 5]
        w_init_interval = [-1.0, 1.0]
        mutation_rate = .1
        prob_node_copy = .1
        init_population = [Entity(n_layer_nodes, w_init_interval)
                           for i in range(self.POP_SIZE)]
        pm = Population_Manager(init_population, mutation_rate, prob_node_copy)
        return pm


if __name__ == '__main__':
    unittest.main()
