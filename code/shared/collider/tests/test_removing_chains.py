from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import unittest
import itertools
import random

import networkx as nx

from .. import collider

def check_graphs_equal(Gtest, Gexpect):
    # # check topologies
    # if not nx.algorithms.isomorphism.DiGraphMatcher(Gtest, Gexpect).is_isomorphic():
    #     raise AssertionError("Graphs not isomorphic")

    # # checks naming
    # ntest = set(Gtest.nodes_iter())
    # nexpect = set(Gexpect.nodes_iter())
    # if ntest - nexpect:
    #     raise AssertionError("Unexpected node present: {}".format(ntest-nexpect))
    # if nexpect - ntest:
    #     raise AssertionError("Expected node missing: {}".format(nexpect-ntest))
    node_mismatch = set()
    edge_mismatch = set()
    for n, (A, B) in enumerate([(Gtest, Gexpect), (Gexpect, Gtest)], start=1):
        for node in A:
            if not B.has_node(node):
                node_mismatch.add(node)
        for edge in A.edges_iter():
            if not B.has_edge(*edge):
                edge_mismatch.add(edge)

    errors = []
    if node_mismatch:
        errors.append('Node mismatch: {}'.format(node_mismatch))
    if edge_mismatch:
        errors.append('Edge mismatch: {}'.format(edge_mismatch))
    if errors:
        raise AssertionError(', '.join(str(e) for e in errors))

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
        graph = nx.DiGraph()
    timepoints = iter(timepoints)

    birth = six.next(timepoints)
    for node_group, death in zip(nodes, timepoints):
        graph.add_nodes_from(node_group, born=birth, died=death)
        birth = death

    return graph

class TestGraphChecking(unittest.TestCase):
    def setUp(self):
        Go = nx.DiGraph()
        Go.add_path([1, 2, 3])
        self.Go = Go

    def test_pass(self):
        Gtest = nx.DiGraph()
        Gtest.add_path([1, 2, 3])
        check_graphs_equal(self.Go, Gtest)

    def test_different_nodes(self):
        Gtest = nx.DiGraph()
        Gtest.add_path([3, 2, 1])
        try:
            check_graphs_equal(self.Go, Gtest)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (renamed nodes)')

    def test_excess_nodes(self):
        Gtest = nx.DiGraph()
        Gtest.add_path([1, 2, 3])
        Gtest.add_node(4)
        try:
            check_graphs_equal(self.Go, Gtest)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (excess nodes in 2nd graph)')

        try:
            check_graphs_equal(Gtest, self.Go)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (excess nodes in 1st graph)')

    def test_excess_edges(self):
        Gtest = nx.DiGraph()
        Gtest.add_path([1, 2, 3])
        Gtest.add_edge(1, 3)
        try:
            check_graphs_equal(self.Go, Gtest)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (excess edges in 2nd graph)')

        try:
            check_graphs_equal(Gtest, self.Go)
        except AssertionError:
            pass
        else:
            self.fail('Graph checker passed bad graph (excess edges in 1st graph)')

class TestDirectDescendents(unittest.TestCase):
    def test_basic_pass(self):
        Go = node_generate(
            [[10, 11], [20], [30], [40, 41]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40])
        Go.add_edge(11, 20)
        Go.add_edge(30, 41)
        Gtest = Go.copy()

        collider.remove_single_descendents(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, (20, 30), 40])
        Gexpect.add_path([11, (20, 30), 41])

        check_graphs_equal(Gtest, Gexpect)

    def test_multi_descendent_abort(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40])
        Go.add_path([11, 20, 31, 40])
        Gtest = Go.copy()

        collider.remove_single_descendents(Gtest)

        check_graphs_equal(Gtest, Go)

    def test_descendent_multiparent_abort(self):
        Go = node_generate(
            [[10, 11], [20, 21], [30], [40, 41]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40])
        Go.add_edge(11, 20)
        Go.add_edge(21, 30)
        Go.add_edge(30, 41)
        Gtest = Go.copy()

        collider.remove_single_descendents(Gtest)

        check_graphs_equal(Gtest, Go)

    def test_consecutive(self):
        Go = node_generate(
            [[10, 11], [20], [30], [40], [50, 51]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_edge(11, 20)
        Go.add_edge(40, 51)
        Gtest = Go.copy()

        collider.remove_single_descendents(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, (20, 40), 50])
        Gexpect.add_path([11, (20, 40), 51])

        check_graphs_equal(Gtest, Gexpect)


class TestFissionFusion(unittest.TestCase):
    def test_basic(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            range(100, 700, 100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, (20, 40), 50])
        Gexpect.add_path([11, (20, 40), 51])

        check_graphs_equal(Gtest, Gexpect)

    def test_linear(self):
        """
        Don't do anything with linear succession, that's not our problem.
        (see remove_single_descendents)
        """
        Go = node_generate(
            [[10], [20], [30]],
            range(100, 500, 100))
        Go.add_path([10, 20, 30])
        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest)

        check_graphs_equal(Gtest, Go)

    def test_component_conservation(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            range(100, 700, 100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest)

        self.assertTrue(Gtest.node[(20, 40)]['components'] == set([20, 31, 30, 40]))

    def test_conditional_false(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            range(100, 700, 100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest, max_frames=50)

        check_graphs_equal(Gtest, Go)

    def test_conditional_true(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            [100, 200, 300, 350, 500, 600])
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])

        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest, max_frames=50)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, (20, 40), 50])
        Gexpect.add_path([11, (20, 40), 51])

        check_graphs_equal(Gtest, Gexpect)

    def test_chain(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51], [60], [70, 71]],
            [100, 200, 300, 400, 500, 600, 700, 800])
        Go.add_path([10, 20, 30, 40, 50, 60, 70])
        Go.add_path([11, 20, 31, 40, 51, 60, 71])
        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, (20, 60), 70])
        Gexpect.add_path([11, (20, 60), 71])

        check_graphs_equal(Gtest, Gexpect)

    def test_chain_rechecking(self):
        """
        If the keys work out just-so that it goes bottom-up, it won't
        actually verify that it recursively checks the new synthetic node
        """
        for seed in range(100):
            random.seed(seed)
            nodes_per_level = [2, 1, 2, 1, 2, 1, 2]
            node_numbers = random.sample(range(int(1e8)), sum(nodes_per_level))
            nodes = [[node_numbers.pop() for _ in range(nn)] for nn in nodes_per_level]
            randomizer = [random.randint(0, int(1e8)) for x in nodes]

            Go = node_generate(nodes, range(100, 900, 100))
            Go.add_path([n[0] for n in nodes])
            Go.add_path([n[-1] for n in nodes])
            Gtest = Go.copy()

            collider.remove_fission_fusion(Gtest)

            Gexpect = nx.DiGraph()
            Gexpect.add_path([nodes[0][0], (nodes[1][0], nodes[-2][0]), nodes[-1][0]])
            Gexpect.add_path([nodes[0][-1], (nodes[1][0], nodes[-2][0]), nodes[-1][-1]])

            check_graphs_equal(Gtest, Gexpect)

    def test_chain_components(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51], [60], [70, 71]],
            [100, 200, 300, 400, 500, 600, 700, 800])
        Go.add_path([10, 20, 30, 40, 50, 60, 70])
        Go.add_path([11, 20, 31, 40, 51, 60, 71])
        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest)

        try:
            self.assertTrue(Gtest.node[(20, 60)]['components'] == set([20, 31, 30, 40, 51, 50, 60]))
        except KeyError:
            self.fail('Node (20, 60) not present in expected output')

    def test_join_after_join(self):
        #"""From ex_id = '20130318_131111', target=930, strangeness happens..."""
        nodes = [[288], [289, 290], [293, 172], [349], [350, 351]]
        Go = node_generate(nodes, range(len(nodes) + 1))
        Go.add_path([288, 289, 293])
        Go.add_path([288, 290, 293, 349, 351])
        Go.add_path([172, 349, 350])
        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([(288, 293), 349, 351])
        Gexpect.add_path([172, 349, 350])

        check_graphs_equal(Gtest, Gexpect)

    def test_child_swap(self):
        """
        This happened:

                 A
                / \
               B   \
                    C   ==>  (A, E)
             D     /
              \   /
               \ /
                E

        It shouldn't.
        """
        Go = node_generate(
            [[10, 11], [20], [29, 30, 31], [40], [50, 51]],
            itertools.count(start=100, step=100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 29])
        Go.add_path([31, 40, 51])
        Gtest = Go.copy()

        collider.remove_fission_fusion(Gtest)

        check_graphs_equal(Gtest, Go)
