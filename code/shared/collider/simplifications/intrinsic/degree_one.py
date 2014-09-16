# -*- coding: utf-8 -*-
"""
Manipulations removing degree-one things (short offshoots and basic chains)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import logging

from ..util import frame_filter

L = logging.getLogger(__name__)

__all__ = [
    'remove_single_descendents',
    'remove_offshoots',
]

def remove_single_descendents(digraph):
    """
    Combine direct descendents (and repetitions thereof) into a single node.

    ..
          ~~~~~
           \|/
            A           ~~~~~
            |            \|/
            |     ==>    A-B
            B            /|\
           /|\          ~~~~~
          ~~~~~

    The hidden data will be attached to the nodes as a set, for example from
    above: ``{A, B}``.
    """
    all_nodes = digraph.nodes()

    while all_nodes:
        node = all_nodes.pop()
        if node not in digraph:
            continue # node was already removed/abridged

        children = set(digraph.successors(node))
        if len(children) != 1:
            continue
        child = children.pop()

        if len(digraph.predecessors(child)) != 1:
            continue

        parents = set(digraph.predecessors(node))
        grandchildren = set(digraph.successors(child))

        digraph.condense_nodes(node, child)

        all_nodes.append(node) # recurse

    # graph is modified in-place

def remove_offshoots(digraph, threshold):
    """
    Remove small dead-ends from *digraph* that last less than *threshold*
    frames.
    """
    all_nodes = digraph.nodes()
    filt = frame_filter(threshold)

    while all_nodes:
        node = all_nodes.pop()
        if digraph.in_degree(node) != 1 or digraph.out_degree(node) != 0:
            continue # topology wrong

        if not filt(digraph, [node]):
            continue # lasts too long

        # add to components of parent then remove node
        parent = digraph.predecessors(node)[0]

        digraph.condense_nodes(parent, node, skip_life_recalc=True)

    # graph is modified in-place
