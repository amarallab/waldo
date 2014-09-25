from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import unittest
import itertools

import networkx as nx

from .test_util import node_generate, GraphCheck, Graph
from .. import remove_offshoots

class TestPruneOffshoots(GraphCheck):
    def setUp(self):
        self.G_basic = node_generate(
            [[10, 11], [20], [30, 31], [40, 41], [50]],
            itertools.count(step=100))
        self.G_basic.add_path([10, 20, 30, 40, 50])
        self.G_basic.add_path([11, 20, 31])
        self.G_basic.add_path([30, 41, 50])
        nx.freeze(self.G_basic)

    def test_basic_threshold_cut(self):
        Gtest = Graph(self.G_basic).copy()
        Gexpect = Graph(self.G_basic).copy()
        Gexpect.remove_node(31)

        remove_offshoots(Gtest, Gtest.lifespan_f(31))
        self.check_graphs_equal(Gtest, Gexpect)

    def test_basic_threshold_ignore(self):
        Gtest = Graph(self.G_basic).copy()

        remove_offshoots(Gtest, Gtest.lifespan_f(31) - 1)
        self.check_graphs_equal(Gtest, self.G_basic)

    def test_multi(self):
        Go = node_generate(
            [[10], [20, 21], [30, 31], [40, 41], [50, 51]],
            itertools.count(step=100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_edges_from([(10, 21), (20, 31), (30, 41), (40, 51)])
        Go.node[50]['died_f'] = 1000
        Gtest = Go.copy()

        Gexpect = Go.copy()
        Gexpect.remove_nodes_from([21, 31, 41, 51])

        remove_offshoots(Gtest, Go.lifespan_f(20))
        self.check_graphs_equal(Gtest, Gexpect)

    def test_recorded_in_parent_components(self):
        Gtest = Graph(self.G_basic).copy()

        remove_offshoots(Gtest, Gtest.lifespan_f(31))
        try:
            self.assertEqual(Gtest.node[20]['components'], set([20, 31]))
        except KeyError:
            self.fail("'components' key missing from node 20")
