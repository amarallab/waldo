from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import unittest

import networkx as nx

from .. import collider


def condition_factory(threshold):
    def conditional(graph, nodes):
        for node in nodes:
            if graph.node[node]['frames'] > threshold:
                return False
        return True
    return conditional


class TestBlobReading(unittest.TestCase):
    def test_basic(self):
        Go = nx.DiGraph()
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        collider.remove_chains(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, '20-40', 50])
        Gexpect.add_path([11, '20-40', 51])

        self.assertTrue(nx.algorithms.isomorphism.DiGraphMatcher(Gtest, Gexpect).is_isomorphic())

    def test_conditional_false(self):
        Go = nx.DiGraph()
        path1, path2 = [10, 20, 30, 40, 50], [11, 20, 31, 40, 51]
        Go.add_nodes_from(path1 + path2, frames=100)
        Go.add_path(path1)
        Go.add_path(path2)
        Gtest = Go.copy()

        # check we actually actually node attribute
        self.assertEqual(Gtest.node[30]['frames'], 100)

        collider.remove_chains(Gtest, conditional=condition_factory(50))

        self.assertTrue(nx.algorithms.isomorphism.DiGraphMatcher(Gtest, Go).is_isomorphic())

    def test_conditional_true(self):
        Go = nx.DiGraph()
        path1, path2 = [10, 20, 30, 40, 50], [11, 20, 31, 40, 51]
        Go.add_nodes_from(path1 + path2, frames=100)
        Go.add_path(path1)
        Go.add_path(path2)

        Gtest = Go.copy()
        Gtest.node[30]['frames'] = 50
        Gtest.node[31]['frames'] = 50

        collider.remove_chains(Gtest, conditional=condition_factory(50))

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, '20-40', 50])
        Gexpect.add_path([11, '20-40', 51])

        self.assertTrue(nx.algorithms.isomorphism.DiGraphMatcher(Gtest, Gexpect).is_isomorphic())

    def test_chain(self):
        Go = nx.DiGraph()
        path1, path2 = [10, 20, 30, 40, 50, 60, 70], [11, 20, 31, 40, 51, 60, 71]
        Go.add_nodes_from(path1 + path2, frames=100)
        Go.add_path(path1)
        Go.add_path(path2)
        Gtest = Go.copy()

        collider.remove_chains(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, '20-60', 70])
        Gexpect.add_path([11, '20-60', 71])

        print(Gexpect.nodes())
        print(Gtest.nodes())
        raise Exception("Check node names are equal too!")
        self.assertTrue(nx.algorithms.isomorphism.DiGraphMatcher(Gtest, Gexpect).is_isomorphic())
