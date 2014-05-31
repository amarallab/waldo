from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools

import networkx as nx

from ..collider import remove_single_descendents
from .test_util import node_generate, GraphCheck

class TestDirectDescendents(GraphCheck):
    def test_basic_pass(self):
        Go = node_generate(
            [[10, 11], [20], [30], [40, 41]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40])
        Go.add_edge(11, 20)
        Go.add_edge(30, 41)
        Gtest = Go.copy()

        remove_single_descendents(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, (20, 30), 40])
        Gexpect.add_path([11, (20, 30), 41])

        self.check_graphs_equal(Gtest, Gexpect)

    def test_multi_descendent_abort(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40])
        Go.add_path([11, 20, 31, 40])
        Gtest = Go.copy()

        remove_single_descendents(Gtest)

        self.check_graphs_equal(Gtest, Go)

    def test_descendent_multiparent_abort(self):
        Go = node_generate(
            [[10, 11], [20, 21], [30], [40, 41]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40])
        Go.add_edge(11, 20)
        Go.add_edge(21, 30)
        Go.add_edge(30, 41)
        Gtest = Go.copy()

        remove_single_descendents(Gtest)

        self.check_graphs_equal(Gtest, Go)

    def test_consecutive(self):
        Go = node_generate(
            [[10, 11], [20], [30], [40], [50, 51]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_edge(11, 20)
        Go.add_edge(40, 51)
        Gtest = Go.copy()

        remove_single_descendents(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, (20, 40), 50])
        Gexpect.add_path([11, (20, 40), 51])

        self.check_graphs_equal(Gtest, Gexpect)
