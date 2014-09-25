from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import random

from nose.tools import nottest
import networkx as nx

from .test_util import node_generate, cumulative_sum, GraphCheck

from .. import assimilate

def graph1():
    """                               ___0
            |12      |10     |11
            |        |       |
            |       _|__     |        ___100
            |    20|    |    |
            |______|    |    |        ___110
               |      21|    |
               |30      |    |
             __|__      |    |        ___120
            |     |     |    |
            |     |     |    |
          40|   41|     |____|        ___150
            |     |       |
            |     |       |42
            |     |       |           ___200
    """
    Go = node_generate(
                [[10, 11, 12], [20, 21], [30], [40, 41, 42]],
                [0, 100, 110, 120, 200])

    Go.node[12]['died_f'] = Go.node[30]['born_f']

    Go.node[11]['died_f'] = Go.node[21]['died_f'] = Go.node[42]['born_f'] = 150

    Go.node[21]['components'] = set([21, 22, 39])
    Go.node[20]['components'] = set([20, 23, 38])

    Go.add_path([12, 30, 40])
    Go.add_path([10, 20, 30, 41])
    Go.add_path([10, 21, 42])
    Go.add_path([11, 42])

    return Go

class TestAssimilator(GraphCheck):
    def threshold_compare(self, Gtest, Gexpect, just_enough):
        Gt1 = Gtest.copy()
        Gt2 = Gtest.copy()
        assimilate(Gt1, max_threshold=just_enough)
        assimilate(Gt2, max_threshold=just_enough - 0.1)

        self.check_graphs_equal(Gexpect, Gt1)
        try:
            self.check_graphs_equal(Gexpect, Gt2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Graphs equal despite threshold too low')

    def test_basic(self):
        Go = node_generate([[10], [20, 21, 22], [30]],
                           cumulative_sum([10, 1, 10]))
        for x in [20, 21, 22]:
            Go.add_path([10, x, 30])
        Gtest = Go.copy()

        Gexpect = node_generate([[10], [30]], cumulative_sum([11, 10]))
        Gexpect.add_path([10, 30])

        self.threshold_compare(Gtest, Gexpect, Go.lifespan_f(20))

    def test_ignorable_topology(self):
        Go = node_generate([[10, 11], [20], [30, 31]],
                           cumulative_sum([10, 1, 10]))
        for start, end in [(10, 30), (11, 31)]:
            Go.add_path([start, 20, end])
        Gtest = Go.copy()

        assimilate(Gtest, Go.lifespan_f(20) + 5)
        self.check_graphs_equal(Go, Gtest)

    def test_preserve_components(self):
        # data in the absorber
        Gtest = graph1()

        assimilate(Gtest, Gtest.lifespan_f(20))

        self.assertTrue('components' in Gtest.node[10])
        self.assertEquals(Gtest.node[10]['components'], set([10, 20, 23, 38]))

    def test_preserve_attributes(self):
        # data in the absorber
        Gtest = graph1()
        Gtest.node[10]['extras'] = {'test': 42}

        assimilate(Gtest, Gtest.lifespan_f(20))

        self.assertTrue('extras' in Gtest.node[10])
        self.assertEquals(Gtest.node[10]['extras'], {'test': 42})

        # data in the absorbed
        Gtest = graph1()
        Gtest.node[20]['extras'] = {'test': 42}

        assimilate(Gtest, Gtest.lifespan_f(20))

        self.assertTrue('extras' in Gtest.node[10])
        self.assertEquals(Gtest.node[10]['extras'], {'test': 42})

    def test_combine_attributes(self):
        Gtest = graph1()
        Gtest.node[10]['extras'] = {'test': 42}
        Gtest.node[20]['extras'] = {'test2': 420}

        assimilate(Gtest, Gtest.lifespan_f(20))

        self.assertTrue('extras' in Gtest.node[10])
        self.assertEquals(Gtest.node[10]['extras'], {'test': 42, 'test2': 420})

    def test_expiration_time(self):
        """'died_f' time should be that of the latest blob in a compound node"""
        Gtest = graph1()

        assimilate(Gtest, Gtest.lifespan_f(20))

        self.assertEqual(Gtest.node[10]['died_f'], 110)

    def test_check_already_removed(self):
        for seed in range(100):
            random.seed(seed)
            nodes_per_level = [2, 1, 2, 1, 2, 1, 2]
            node_numbers = sorted(random.sample(range(int(1e8)), sum(nodes_per_level)))
            nodes = [[node_numbers.pop() for _ in range(nn)] for nn in nodes_per_level]
            randomizer = [random.randint(0, int(1e8)) for x in nodes]

            small_node_frames = snf = 10
            small_node_lifespan = small_node_frames + 1 # fencepost

            Go = node_generate(nodes, cumulative_sum([100, snf, snf, snf, snf, snf, 100]))
            Go.add_path([n[0] for n in nodes])
            Go.add_path([n[-1] for n in nodes])
            Gtest = Go.copy()

            try:
                assimilate(Gtest, small_node_lifespan)
            except nx.NetworkXError as e:
                self.fail('NetworkX Error, likely from looking at a node already removed:\n\t{}'.format(e))

    def test_node_name_shuffling(self):
        for seed in range(100):
            random.seed(seed)
            nodes_per_level = [2, 1, 2, 1, 2, 1, 2]
            node_numbers = sorted(random.sample(range(int(1e8)), sum(nodes_per_level)))
            nodes = [[node_numbers.pop() for _ in range(nn)] for nn in nodes_per_level]
            randomizer = [random.randint(0, int(1e8)) for x in nodes]

            small_node_frames = snf = 10
            small_node_lifespan = small_node_frames + 1 # fencepost

            Go = node_generate(nodes, cumulative_sum([100, snf, snf, snf, snf, snf, 100]))
            Go.add_path([n[0] for n in nodes])
            Go.add_path([n[-1] for n in nodes])
            Gtest = Go.copy()

            assimilate(Gtest, small_node_lifespan)

            Gexpect = nx.DiGraph()
            Gexpect.add_path([nodes[0][0], nodes[1][0], nodes[-1][0]])
            Gexpect.add_path([nodes[0][-1], nodes[1][0], nodes[-1][-1]])

            print('Expected:', sorted(list(Gexpect.edges())))
            print('Test:', sorted(list(Gtest.edges())))

            self.check_graphs_equal(Gtest, Gexpect)
