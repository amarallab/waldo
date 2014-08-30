# -*- coding: utf-8 -*-
"""
MWT graph manipulations that remove spurious fission-fusion events
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools
import collections
try:
    from statistics import mean # stdlib 3.4+
except ImportError:
    def mean(x):
        return sum(x) / len(x)

from ..util import frame_filter

__all__ = [
    'remove_fission_fusion',
    'remove_fission_fusion_rel',
]

def remove_fission_fusion(digraph, max_split_frames=None):
    """
    Strip out fission-fusion events (and repetitions thereof) from the
    graph.

    ..
           ~~~
            |
            A
           / \           ~~~
          /   \           |
         B     C   ==>   A-D
          \   /           |
           \ /           ~~~
            D
            |
           ~~~

    The hidden data will be attached to the nodes as a set, for example from
    above: ``{A, B, C, D}``.

    """
    if max_split_frames is None:
        conditional = None
    else:
        conditional = frame_filter(max_split_frames)

    all_nodes = digraph.nodes()

    while all_nodes:
        node = all_nodes.pop()
        if node not in digraph:
            continue # node was already removed/abridged

        parents = set(digraph.predecessors(node))
        children = set(digraph.successors(node))

        if len(children) != 2:
            continue # no fission occured
            # (probably a job for remove_single_descendents())

        grandchildren = set()
        abort = False
        for child in children:
            new_gc = digraph.successors(child)
            if len(new_gc) != 1:
                abort = True
                break
            grandchildren.update(new_gc)
        if abort:
            continue # see TestFissionFusion.test_child_swap

        if len(grandchildren) != 1:
            continue # skip doing anything

        if conditional is not None:
            if not conditional(digraph, children):
                continue # children fail conditional testing

        grandchild = grandchildren.pop()
        greatgrandchildren = set(digraph.successors(grandchild))

        digraph.condense_nodes(node, *(children | set([grandchild])))

        all_nodes.append(node) # recurse

    # graph is modified in-place

def remove_fission_fusion_rel(digraph, split_rel_time):
    """
    Strip out fission-fusion events (and repetitions thereof) from the
    digraph.

    ..
           ~~~
            |
            A
           / \           ~~~
          /   \           |
         B     C   ==>   A-D
          \   /           |
           \ /           ~~~
            D
            |
           ~~~

    The hidden data will be attached to the nodes as a set, for example from
    above: ``{A, B, C, D}``.

    """
    def conditional(digraph, node, children, grandchild):
        # average age of focal node/gchild
        endpoint_avg = mean([digraph.lifespan_f(node),
                        digraph.lifespan_f(grandchild)])
        # average age of children
        split_avg = mean([digraph.lifespan_f(c) for c in children])

        return split_avg / endpoint_avg <= split_rel_time

    all_nodes = digraph.nodes()
    # order matters here, random key hashes...yadda yadda.
    all_nodes.sort(key=lambda x: x[1] if isinstance(x, tuple) else x)

    while all_nodes:
        node = all_nodes.pop() # pop latest node, work end to start
        if node not in digraph:
            continue # node was already removed/abridged

        parents = set(digraph.predecessors(node))
        children = set(digraph.successors(node))

        if len(children) != 2:
            continue # no fission occured
            # (probably a job for remove_single_descendents())

        grandchildren = set()
        abort = False
        for child in children:
            new_gc = digraph.successors(child)
            if len(new_gc) != 1:
                abort = True
                break
            grandchildren.update(new_gc)
        if abort:
            continue # see TestFissionFusion.test_child_swap

        if len(grandchildren) != 1:
            continue # skip doing anything

        grandchild = grandchildren.pop()

        if not conditional(digraph, node, children, grandchild):
            continue # failed conditional testing

        greatgrandchildren = set(digraph.successors(grandchild))

        digraph.condense_nodes(node, *(children | set([grandchild])))

        all_nodes.append(node) # recurse

    # graph is modified in-place
