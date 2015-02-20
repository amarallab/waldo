from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import random
import itertools

from nose.tools import nottest
import networkx as nx

from waldo.network.tests import test_graph as tg
from waldo.network import Graph

from .test_util import cumulative_sum

from .. import collapse_group_of_nodes

class TestCollapseGroupOfNodes(tg.GraphTestCase):
    def test_diamond(self):
        Go = tg.node_generate([[10], [20, 21], [30]],
                           itertools.count(100))

        Go.add_path([10, 20, 30])
        Go.add_path([10, 21, 30])

        Gexpect = tg.node_generate([[10]])

        max_duration = 250
        print(Go.edges())
        print(max_duration)

        collapse_group_of_nodes(Go, max_duration)

        self.assertTopologyEqual(Go, Gexpect)

    @nottest
    def test_basic_with_tail(self):
        Go = tg.node_generate([[10], [20, 21], [30], [40]],
                           itertools.count(100))

        Go.add_path([10, 20, 30, 40])
        Go.add_path([10, 21, 30])

        Gexpect = tg.node_generate([[10], [40]])
        Gexpect.add_path([10, 40])

        #def collapse_group_of_nodes(graph, max_duration, verbose=False):
        max_duration = 250
        print(Go.edges())
        #print(Go.node[20]['born_f'])
        print(max_duration)

        collapse_group_of_nodes(Go, max_duration)

        self.assertTopologyEqual(Go, Gexpect)

    @nottest
    def test_double_diamond(self):
        Go = tg.node_generate([[10], [20, 21], [30], [40, 41], [50]],
                           itertools.count(100))

        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([10, 21, 30, 41, 50])

        Gtest = Go.copy()

        Gexpect = tg.node_generate([[10], [40, 41], [50]], [0, 300, 400, 500])
        Gexpect.add_path([10, 41, 50])
        Gexpect.add_path([10, 40, 50])

        #def collapse_group_of_nodes(graph, max_duration, verbose=False):
        max_duration = 250 * tg.FRAME_TIME
        print(Gtest.node[20]['born_t'])
        print(Gtest.node[20]['born_f'])
        print(max_duration)

        collapse_group_of_nodes(Gtest, max_duration)

        self.assertTopologyEqual(Gtest, Gexpect)
