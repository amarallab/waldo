from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import unittest

import networkx as nx

from .test_util import node_generate, GraphCheck
from ..simplifications.util import condense_nodes

class TestNodeCondensing(GraphCheck):
    def test_topology(self):
        Go = node_generate([[10], [20], [30]], [0, 100, 200, 300])
        Gtest = Go.copy()

        Gexpect = node_generate([[10], [30]], [0, 200, 300])

        condense_nodes(Gtest, 10, 20)

        self.check_graphs_equal(Gtest, Gexpect)

    def test_component_storage(self):
        Go = node_generate([[10], [20], [30]], [0, 100, 200, 300])
        Gtest = Go.copy()

        condense_nodes(Gtest, 10, 20)

        self.assertEquals(Gtest.node[10]['components'], set([10, 20]))

    def test_component_transfer(self):
        Go = node_generate([[10], [20], [30]], [0, 100, 200, 300])
        Go.node[20]['components'] = set([20, 21, 22])
        Gtest = Go.copy()

        condense_nodes(Gtest, 10, 20)

        self.assertEquals(Gtest.node[10]['components'], set([10, 20, 21, 22]))

    def test_born_died(self):
        Go = node_generate([[10], [20], [30]], [0, 100, 200, 300])
        Gtest = Go.copy()

        condense_nodes(Gtest, 10, 20)

        self.assertEquals(Gtest.node[10]['died'], 200)
