# -*- coding: utf-8 -*-
"""
Remove nodes that have no edges and no position data.
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

__all__ = [
    'remove_blank_nodes',
]

def remove_blank_nodes(digraph, experiment):
    """
    Remove unconnected nodes in *digraph* with no associated data in
    *experiment*.
    """
    for bid in experiment:
        if bid in digraph and not digraph.neighbors(bid):
            if experiment[bid].empty:
                digraph.remove_node(bid)
