# -*- coding: utf-8 -*-
"""
Manipulations removing degree-one things (short offshoots and basic chains)
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

from ..util import frame_filter, condense_nodes

__all__ = [
    'remove_single_descendents',
    'remove_offshoots',
]

def remove_single_descendents(graph):
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
    all_nodes = graph.nodes()

    while all_nodes:
        node = all_nodes.pop()
        if node not in graph:
            continue # node was already removed/abridged

        children = set(graph.successors(node))
        if len(children) != 1:
            continue
        child = children.pop()

        if len(graph.predecessors(child)) != 1:
            continue

        parents = set(graph.predecessors(node))
        grandchildren = set(graph.successors(child))

        new_node, new_node_data = condense_nodes(graph, node, child)

        graph.add_node(new_node, **new_node_data)
        graph.add_edges_from((p, new_node) for p in parents)
        graph.add_edges_from((new_node, gc) for gc in grandchildren)
        graph.remove_nodes_from([node, child])

        all_nodes.append(new_node)

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

        _, new_node_data = condense_nodes(digraph, parent, parent, node)
        digraph.node[parent] = new_node_data

        digraph.remove_node(node)

    # graph is modified in-place
