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

def remove_blank_nodes(digraph, experiment, exclude_connected=True):
    """
    Remove nodes in *digraph* with no associated data in *experiment*. If
    *exclude_connected*, skip nodes that are connected.

    Returns a set containing removed blobs
    """
    removed = set()
    for bid in digraph.nodes():
        if not (exclude_connected and digraph.neighbors(bid)):
            if experiment[bid].empty:
                digraph.remove_node(bid)
                removed.add(bid)

    return removed
