from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import networkx as nx

from waldo.network.tests import test_graph as tg

from waldo.network import Graph

class TestNodeCondensing(tg.GraphTestCase):
    def test_topology(self):
        Go = tg.node_generate([[10], [20], [30]], [0, 100, 200, 300])
        Go.add_path([10, 20, 30])
        Gtest = Graph(Go)

        Gexpect = tg.node_generate([[10], [30]], [0, 200, 300])
        Gexpect.add_path([10, 30])

        Gtest.condense_nodes(10, 20)

        self.assertTopologyEqual(Gtest, Gexpect)

    def test_component_storage(self):
        Go = tg.node_generate([[10], [20], [30]], [0, 100, 200, 300])
        Go.add_path([10, 20, 30])
        Gtest = Graph(Go)

        Gtest.condense_nodes(10, 20)

        self.assertEquals(Gtest.node[10]['components'], set([10, 20]))

    def test_component_transfer(self):
        Go = tg.node_generate([[10], [20], [30]], [0, 100, 200, 300])
        Go.add_path([10, 20, 30])
        Go.node[20]['components'] = set([20, 21, 22])
        Gtest = Graph(Go)

        Gtest.condense_nodes(10, 20)

        self.assertEquals(Gtest.node[10]['components'], set([10, 20, 21, 22]))

    def test_born_died(self):
        Go = tg.node_generate([[10], [20], [30]], [0, 100, 200, 300])
        Go.add_path([10, 20, 30])
        Gtest = Graph(Go)

        Gtest.condense_nodes(10, 20)

        self.assertEquals(Gtest.node[10]['died_f'], 200)
