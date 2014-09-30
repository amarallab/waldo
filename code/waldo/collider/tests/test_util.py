from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import unittest

from waldo.network import Graph

FRAME_TIME = 1/8

def node_generate(nodes, timepoints, graph=None):
    """
    *nodes* is an iterable of iterables that define what nodes to create
    spanning each gap in *timepoints*, an iterable of numbers.  *timepoints*
    must be 1 longer than nodes.
    """
    len_nodes = len(nodes)
    try:
        if len_nodes != len(timepoints) - 1:
            raise ValueError('nodes is not one less in length than timepoints')
    except TypeError:
        # assuming timepoints is a generator which may or may not be
        # limitless
        pass

    if graph is None:
        graph = Graph()
    timepoints = iter(timepoints)

    birth = next(timepoints)
    for node_group, death in zip(nodes, timepoints):
        graph.add_nodes_from(node_group, born_f=birth, died_f=death, born_t=birth*FRAME_TIME, died_t=death*FRAME_TIME)
        birth = death

    return graph

def cumulative_sum(seq, start=0):
    x = start
    yield x
    for element in seq:
        x += element
        yield x


class GraphCheck(unittest.TestCase):
    def check_graphs_equal(self, Gtest, Gexpect):
        # # check topologies
        nGt = set(Gtest.nodes_iter())
        nGe = set(Gexpect.nodes_iter())
        eGt = set(Gtest.edges_iter())
        eGe = set(Gexpect.edges_iter())
        self.assertTrue(nGt == nGe and eGt == eGe, 'Graph mismatch.\n\n'
                'Unexpected nodes:        {}\n'
                'Expected, missing nodes: {}\n\n'
                'Unexpected edges:        {}\n'
                'Expected, missing edges: {}'.format(
                        ', '.join(str(x) for x in nGt-nGe),
                        ', '.join(str(x) for x in nGe-nGt),
                        ', '.join(str(x) for x in eGt-eGe),
                        ', '.join(str(x) for x in eGe-eGt)))


class TestGraphChecking(GraphCheck):
    def setUp(self):
        Go = Graph()
        Go.add_path([1, 2, 3])
        self.Go = Go

    def test_pass(self):
        Gtest = Graph()
        Gtest.add_path([1, 2, 3])
        self.check_graphs_equal(self.Go, Gtest)

    def test_different_nodes(self):
        Gtest = Graph()
        Gtest.add_path([3, 2, 1])
        try:
            self.check_graphs_equal(self.Go, Gtest)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (renamed nodes)')

    def test_excess_nodes(self):
        Gtest = Graph()
        Gtest.add_path([1, 2, 3])
        Gtest.add_node(4)
        try:
            self.check_graphs_equal(self.Go, Gtest)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (excess nodes in 2nd graph)')

        try:
            self.check_graphs_equal(Gtest, self.Go)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (excess nodes in 1st graph)')

    def test_excess_edges(self):
        Gtest = Graph()
        Gtest.add_path([1, 2, 3])
        Gtest.add_edge(1, 3)
        try:
            self.check_graphs_equal(self.Go, Gtest)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (excess edges in 2nd graph)')

        try:
            self.check_graphs_equal(Gtest, self.Go)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (excess edges in 1st graph)')