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
    def test_diamond_shouldcondense(self):
        Gtest = tg.node_generate(
                [[10], [20, 21], [30]],
                itertools.count(100))
        Gtest.add_path([10, 20, 30])
        Gtest.add_path([10, 21, 30])

        # result: single node graph
        Gexpect = tg.node_generate([[10]])

        max_duration = 110 * tg.FRAME_TIME

        collapse_group_of_nodes(Gtest, max_duration)

        self.assertTopologyEqual(Gtest, Gexpect)

    def test_diamond_toolarge(self):
        Gtest = tg.node_generate(
                [[10], [20, 21], [30]],
                itertools.count(100))
        Gtest.add_path([10, 20, 30])
        Gtest.add_path([10, 21, 30])

        # no change
        Gexpect = Gtest.copy()

        max_duration = 90 * tg.FRAME_TIME

        collapse_group_of_nodes(Gtest, max_duration)

        self.assertTopologyEqual(Gtest, Gexpect)

    def test_diamondwithtail_shouldcondense(self):
        Gtest = tg.node_generate(
                [[10], [20, 21], [30], [40]],
                cumulative_sum([100, 100, 100, 1000]))
        Gtest.add_path([10, 20, 30, 40])
        Gtest.add_path([10, 21, 30])

        Gexpect = tg.node_generate([[10], [40]])
        Gexpect.add_path([10, 40])

        max_duration = 110 * tg.FRAME_TIME

        collapse_group_of_nodes(Gtest, max_duration)

        self.assertTopologyEqual(Gtest, Gexpect)

    def test_diamondwithtwotails_shouldcondense(self):
        Gtest = tg.node_generate(
                [[10], [20, 21], [30], [40, 41]],
                cumulative_sum([100, 100, 100, 1000]))
        Gtest.add_path([10, 20, 30, 40])
        Gtest.add_path([10, 21, 30, 41])

        Gexpect = tg.node_generate([[10], [40, 41]])
        Gexpect.add_path([10, 40])
        Gexpect.add_path([10, 41])

        max_duration = 110 * tg.FRAME_TIME

        collapse_group_of_nodes(Gtest, max_duration)

        self.assertTopologyEqual(Gtest, Gexpect)

    def test_doublediamond_condensefirst(self):
        Gtest = tg.node_generate(
                [[10], [20, 21], [30], [40, 41], [50]],
                cumulative_sum([100, 100, 100, 200, 100]))
        Gtest.add_path([10, 20, 30, 40, 50])
        Gtest.add_path([10, 21, 30, 41, 50])

        Gexpect = tg.node_generate([[10], [40, 41], [50]])
        Gexpect.add_path([10, 41, 50])
        Gexpect.add_path([10, 40, 50])

        # The first diamond is 100 frames long in middle, plus 100 for the
        # end, so this should encompass it. The second is 200 + 100 so
        # shouldn't.
        max_duration = 110 * tg.FRAME_TIME

        collapse_group_of_nodes(Gtest, max_duration)

        self.assertTopologyEqual(Gtest, Gexpect)

    def test_doublediamond_condenseboth(self):
        Gtest = tg.node_generate(
                [[10], [20, 21], [30], [40, 41], [50]],
                itertools.count(100))
        Gtest.add_path([10, 20, 30, 40, 50])
        Gtest.add_path([10, 21, 30, 41, 50])

        # result: single node graph
        Gexpect = tg.node_generate([[10]])

        # the first diamond is 200 frames long, second also 200, so this
        # should encompass both.
        max_duration = 310 * tg.FRAME_TIME

        collapse_group_of_nodes(Gtest, max_duration)

        self.assertTopologyEqual(Gtest, Gexpect)

    def test_exchange_shouldcondense(self):
        Gtest = tg.node_generate(
                [[10], [20, 21], [30, 31, 32], [40, 41], [50]],
                itertools.count(100))
        Gtest.add_path([10, 21, 30, 40, 50])
        Gtest.add_path([10, 20, 31, 40])
        Gtest.add_path([21, 32, 41, 50])
        Gtest.condense_nodes(20, 31)
        Gtest.condense_nodes(21, 32)

        Gexpect = tg.node_generate([[10]])

        max_duration = 310 * tg.FRAME_TIME

        collapse_group_of_nodes(Gtest, max_duration)

        self.assertTopologyEqual(Gtest, Gexpect)
