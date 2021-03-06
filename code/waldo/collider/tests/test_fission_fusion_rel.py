from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import unittest
import itertools
import random

import networkx as nx

from waldo.network.tests import test_graph as tg
from waldo.network import Graph

from .test_util import cumulative_sum

from .. import remove_fission_fusion_rel

class TestFissionFusion(tg.GraphTestCase):
    def threshold_compare(self, Gtest, Gexpect, just_enough):
        Gt1 = Gtest.copy()
        Gt2 = Gtest.copy()
        remove_fission_fusion_rel(Gt1, split_rel_time=just_enough)
        remove_fission_fusion_rel(Gt2, split_rel_time=just_enough * 0.99999)

        self.assertTopologyEqual(Gexpect, Gt1)
        try:
            self.assertTopologyEqual(Gexpect, Gt2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Graphs equal despite threshold too low')

    def test_basic(self):
        Go = tg.node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            [0, 1, 11, 12, 22, 23])
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, 20, 50])
        Gexpect.add_path([11, 20, 51])

        self.threshold_compare(
                Gtest,
                Gexpect,
                Go.lifespan_f(30) / Go.lifespan_f(20))

    def test_linear(self):
        """
        Don't do anything with linear succession, that's not our problem.
        (see remove_single_descendents)
        """
        Go = tg.node_generate(
            [[10], [20], [30]],
            range(100, 500, 100))
        Go.add_path([10, 20, 30])
        Gtest = Go.copy()

        remove_fission_fusion_rel(Gtest, 1e100)

        self.assertTopologyEqual(Gtest, Go)

    def test_component_conservation(self):
        Go = tg.node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51]],
            [0, 1, 11, 12, 22, 23])
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 31, 40, 51])
        Gtest = Go.copy()

        remove_fission_fusion_rel(Gtest, Go.lifespan_f(30) / Go.lifespan_f(20))

        try:
            self.assertTrue(Gtest.node[20]['components'] == set([20, 31, 30, 40]))
        except KeyError:
            self.fail('Expected node 20 not present in output')

    def test_chain(self):
        Go = tg.node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51], [60], [70, 71]],
            [100, 200, 300, 400, 500, 600, 700, 800])
        Go.add_path([10, 20, 30, 40, 50, 60, 70])
        Go.add_path([11, 20, 31, 40, 51, 60, 71])
        Gtest = Go.copy()

        Gexpect = nx.DiGraph()
        Gexpect.add_path([10, 20, 70])
        Gexpect.add_path([11, 20, 71])

        just_enough = 1
        self.threshold_compare(Gtest, Gexpect, just_enough)

    # def test_component_conservation_chain(self):
    #     Go = tg.node_generate(
    #         [[10, 11], [20], [30, 31], [40], [50, 51], [60], [70, 71]],
    #         itertools.count(step=100))
    #     Go.add_path([10, 20, 30, 40, 50, 60, 70])
    #     Go.add_path([11, 20, 31, 40, 51, 60, 71])
    #     Gtest = Go.copy()

    #     remove_fission_fusion(Gtest)
    #     self.assertEqual(
    #         Gtest.node[(20, 60)]['components'],
    #         set([20, 31, 30, 40, 50, 51, 60]))

    def test_chain_rechecking(self):
        """
        If the keys work out just-so that it goes bottom-up, it won't
        actually verify that it recursively checks the new synthetic node
        """
        for seed in range(100):
            random.seed(seed)
            print('Seed: {}...'.format(seed), end=' ')
            nodes_per_level = [2, 1, 2, 1, 2, 1, 1]
            node_numbers = random.sample(range(int(1e8)), sum(nodes_per_level))
            node_numbers.sort()
            nodes = [[node_numbers.pop(0) for _ in range(nn)] for nn in nodes_per_level]

            Go1 = tg.node_generate(nodes, cumulative_sum([100, 100, 101, 100, 100, 100, 100]))
            Go1.add_path([n[0] for n in nodes])
            Go1.add_path([n[-1] for n in nodes])
            Gtest1 = Go1.copy()

            Go2 = tg.node_generate(nodes, cumulative_sum([100, 100, 100, 100, 101, 100, 100]))
            Go2.add_path([n[0] for n in nodes])
            Go2.add_path([n[-1] for n in nodes])
            Gtest2 = Go2.copy()

            Gexpect = nx.DiGraph()
            Gexpect.add_path([nodes[0][0], nodes[1][0], nodes[-1][0]])
            Gexpect.add_path([nodes[0][-1], nodes[1][0], nodes[-1][-1]])

            remove_fission_fusion_rel(Gtest1, 1)
            remove_fission_fusion_rel(Gtest2, 1)
            self.assertTopologyEqual(Gtest1, Gexpect)
            self.assertTopologyEqual(Gtest2, Gexpect)

    def test_chain_components(self):
        Go = tg.node_generate(
            [[10, 11], [20], [30, 31], [40], [50, 51], [60], [70, 71]],
            [100, 200, 300, 400, 500, 600, 700, 800])
        Go.add_path([10, 20, 30, 40, 50, 60, 70])
        Go.add_path([11, 20, 31, 40, 51, 60, 71])
        Gtest = Go.copy()

        remove_fission_fusion_rel(Gtest, 1e100)

        try:
            self.assertTrue(Gtest.node[20]['components'] == set([20, 31, 30, 40, 51, 50, 60]))
        except KeyError:
            self.fail('Expected node 20 not present in output')

    # def test_join_after_join(self):
    #     #"""From ex_id = '20130318_131111', target=930, strangeness happens..."""
    #     nodes = [[288], [289, 290], [293, 172], [349], [350, 351]]
    #     Go = tg.node_generate(nodes, range(len(nodes) + 1))
    #     Go.add_path([288, 289, 293])
    #     Go.add_path([288, 290, 293, 349, 351])
    #     Go.add_path([172, 349, 350])
    #     Gtest = Go.copy()

    #     remove_fission_fusion(Gtest)

    #     Gexpect = nx.DiGraph()
    #     Gexpect.add_path([(288, 293), 349, 351])
    #     Gexpect.add_path([172, 349, 350])

    #     self.assertTopologyEqual(Gtest, Gexpect)

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
        Go = tg.node_generate(
            [[10, 11], [20], [29, 30, 31], [40], [50, 51]],
            itertools.count(step=100))
        Go.add_path([10, 20, 30, 40, 50])
        Go.add_path([11, 20, 29])
        Go.add_path([31, 40, 51])
        Gtest = Go.copy()

        remove_fission_fusion_rel(Gtest, 1e100)

        self.assertTopologyEqual(Gtest, Go)

    def _nested_gen(self, times):
        Go = tg.node_generate(
            [[0], [10], [20, 21], [30, 31, 32, 33], [40, 41], [50], [60]],
            times)
        Go.add_path([0, 10, 20, 30, 40, 50, 60])
        Go.add_path([10, 21, 32, 41, 50])
        Go.add_path([21, 33, 41])
        Go.add_path([20, 31, 40])
        return Go

    def test_nested_all_small_enough(self):
        Go = self._nested_gen(cumulative_sum([99, 3, 1, 1, 1, 3, 99]))
        Gtest = Go.copy()

        Gexpect = nx.DiGraph()
        Gexpect.add_path([0, 10, 60])

        just_enough = 1
        self.threshold_compare(Gtest, Gexpect, just_enough)

    def test_nested_branches_too_big(self):
        """
        Main path splits, then splits again.  The two fission-fusion pairs
        along each side are condensed, but then the condensed nodes are too
        big to recursively combine.
        """
        Go = self._nested_gen(itertools.count(step=100))
        Gtest = Go.copy()

        Gexpect = nx.DiGraph()
        Gexpect.add_path([0, 10, 20, 50, 60])
        Gexpect.add_path([10, 21, 50])

        just_enough = 1
        self.threshold_compare(Gtest, Gexpect, just_enough)

    def test_nested_all_too_big(self):
        Go = self._nested_gen(itertools.count(step=100))
        Gtest = Go.copy()

        remove_fission_fusion_rel(Gtest, 0.999)

        self.assertTopologyEqual(Gtest, Go)
