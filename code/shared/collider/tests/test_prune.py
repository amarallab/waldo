from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import unittest

import networkx as nx

from ..collider import remove_offshoots
from .test_util import node_generate, GraphCheck

class TestPruneOffshoots(GraphCheck):
    def test_basic_pass(self):
        pass
