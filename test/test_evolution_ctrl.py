from src.evolutionCtrl import Population_Manager
from src.entity import Entity
import unittest


class Test_Population_Manager(unittest.TestCase):
    def test_ctor(self):
        n_layer_nodes = [5, 5, 5]
        w_init_interval = [-1.0, 1.0]
        pop_size = 10
        mutation_rate = .1
        prob_node_copy = .1
        init_population = [Entity(n_layer_nodes, w_init_interval)
                           for i in range(pop_size)]
        pm = Population_Manager(init_population, mutation_rate, prob_node_copy)
        self.assertEqual(pm.pop_size, pop_size)


if __name__ == '__main__':
    unittest.main()
