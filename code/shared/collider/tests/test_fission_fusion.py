from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import unittest
import itertools
import random

import networkx as nx

from .test_util import node_generate, GraphCheck
from .. import remove_fission_fusion

class TestFissionFusion(GraphCheck):
    def threshold_compare(self, Gtest, Gexpect, just_enough):
        Gt1 = Gtest.copy()
        Gt2 = Gtest.copy()
        remove_fission_fusion(Gt1, max_split_frames=just_enough)
        remove_fission_fusion(Gt2, max_split_frames=just_enough - 0.1)

        self.check_graphs_equal(Gexpect, Gt1)
        try:
            self.check_graphs_equal(Gexpect, Gt2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Graphs equal despite threshold too low')

    def test_basic(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            range(100, 700, 100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        remove_fission_fusion(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, 20, 50])
        Gexpect.add_path([11, 20, 51])

        self.check_graphs_equal(Gtest, Gexpect)

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

        remove_fission_fusion(Gtest)

        self.check_graphs_equal(Gtest, Go)

    def test_component_conservation(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            range(100, 700, 100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        remove_fission_fusion(Gtest)

        try:
            self.assertTrue(Gtest.node[20]['components'] == set([20, 31, 30, 40]))
        except KeyError:
            self.fail('Expected node 20 not present in output')

    def test_conditional_false(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            range(100, 700, 100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        remove_fission_fusion(Gtest, max_split_frames=50)

        self.check_graphs_equal(Gtest, Go)

    def test_conditional_true(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            [100, 200, 300, 350, 500, 600])
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])

        Gtest = Go.copy()

        remove_fission_fusion(Gtest, max_split_frames=50)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, 20, 50])
        Gexpect.add_path([11, 20, 51])

        self.check_graphs_equal(Gtest, Gexpect)

    def test_chain(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51], [60], [70, 71]],
            [100, 200, 300, 400, 500, 600, 700, 800])
        Go.add_path([10, 20, 30, 40, 50, 60, 70])
        Go.add_path([11, 20, 31, 40, 51, 60, 71])
        Gtest = Go.copy()

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, 20, 70])
        Gexpect.add_path([11, 20, 71])

        just_enough = 100
        self.threshold_compare(Gtest, Gexpect, just_enough)

    def test_component_conservation_chain(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51], [60], [70, 71]],
            itertools.count(100))
        Go.add_path([10, 20, 30, 40, 50, 60, 70])
        Go.add_path([11, 20, 31, 40, 51, 60, 71])
        Gtest = Go.copy()

        remove_fission_fusion(Gtest)
        try:
            self.assertEqual(
                Gtest.node[20]['components'],
                set([20, 31, 30, 40, 50, 51, 60]))
        except KeyError:
            self.fail('Expected node 20 not present in output')

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

            remove_fission_fusion(Gtest)

            Gexpect = nx.DiGraph()
            Gexpect.add_path([nodes[0][0], nodes[1][0], nodes[-1][0]])
            Gexpect.add_path([nodes[0][-1], nodes[1][0], nodes[-1][-1]])

            self.check_graphs_equal(Gtest, Gexpect)

    def test_chain_components(self):
        Go = node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51], [60], [70, 71]],
            [100, 200, 300, 400, 500, 600, 700, 800])
        Go.add_path([10, 20, 30, 40, 50, 60, 70])
        Go.add_path([11, 20, 31, 40, 51, 60, 71])
        Gtest = Go.copy()

        remove_fission_fusion(Gtest, 100)

        try:
            self.assertTrue(Gtest.node[20]['components'] == set([20, 31, 30, 40, 51, 50, 60]))
        except KeyError:
            self.fail('Expected node 20 not present in output')

    def test_join_after_join(self):
        #"""From ex_id = '20130318_131111', target=930, strangeness happens..."""
        nodes = [[288], [289, 290], [293, 172], [349], [350, 351]]
        Go = node_generate(nodes, range(len(nodes) + 1))
        Go.add_path([288, 289, 293])
        Go.add_path([288, 290, 293, 349, 351])
        Go.add_path([172, 349, 350])
        Gtest = Go.copy()

        remove_fission_fusion(Gtest)

        Gexpect = nx.DiGraph()
        Gexpect.add_path([288, 349, 351])
        Gexpect.add_path([172, 349, 350])

        self.check_graphs_equal(Gtest, Gexpect)

    def test_child_swap(self):
        """
        This happened:

                A
               / \
              B   \
       ~~~~~       \
        \ /         C   ==>  (A, E)
         D         /
          \____   /
               \ /
                E

        It shouldn't.
        """
        Go = node_generate(
            [[10, 11], [20], [29, 30, 31], [40], [50, 51]],
            itertools.count(step=100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 29])
        Go.add_path([31, 40, 51])
        Gtest = Go.copy()

        remove_fission_fusion(Gtest, 200)

        self.check_graphs_equal(Gtest, Go)

    def _nested_gen(self, times):
        Go = node_generate(
            [[0], [10], [20, 21], [30, 31, 32, 33], [40, 41], [50], [60]],
            times)
        Go.add_path([0, 10, 20, 30, 40, 50, 60])
        Go.add_path([10, 21, 32, 41, 50])
        Go.add_path([21, 33, 41])
        Go.add_path([20, 31, 40])
        return Go

    def test_nested_ff_small_enough(self):
        Go = self._nested_gen(itertools.count(step=100))
        Gtest = Go.copy()

        Gexpect = nx.DiGraph()
        Gexpect.add_path([0, 10, 60])

        just_enough = 300
        self.threshold_compare(Gtest, Gexpect, just_enough)

    def test_nested_ff_some_too_big(self):
        Go = self._nested_gen(itertools.count(step=100))
        Gtest = Go.copy()

        Gexpect = nx.DiGraph()
        Gexpect.add_path([0, 10, 20, 50, 60])
        Gexpect.add_path([10, 21, 50])

        just_enough = 100
        self.threshold_compare(Gtest, Gexpect, just_enough)

    def test_nested_ff_all_too_big(self):
        Go = self._nested_gen(itertools.count(step=100))
        Gtest = Go.copy()

        remove_fission_fusion(Gtest, 99)

        self.check_graphs_equal(Gtest, Go)
