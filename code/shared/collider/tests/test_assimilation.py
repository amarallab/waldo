from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools

import networkx as nx

from .test_util import node_generate, cumulative_sum, GraphCheck

from .. import assimilate

class TestAssimilator(GraphCheck):
    def threshold_compare(self, Gtest, Gexpect, just_enough):
        Gt1 = Gtest.copy()
        Gt2 = Gtest.copy()
        assimilate(Gt1, max_threshold=just_enough)
        assimilate(Gt2, max_threshold=just_enough - 0.1)

        self.check_graphs_equal(Gexpect, Gt1)
        try:
            self.check_graphs_equal(Gexpect, Gt2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Graphs equal despite threshold too low')

    def test_basic(self):
        Go = node_generate([[10], [20, 21, 22], [30]],
                           cumulative_sum([10, 1, 10]))
        for x in [20, 21, 22]:
            Go.add_path([10, x, 30])
        Gtest = Go.copy()

        Gexpect = node_generate([[10], [30]], cumulative_sum([11, 10]))

        self.threshold_compare(Gtest, Gexpect, 1)

    def test_other_parents(self):
        pass