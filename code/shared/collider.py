# -*- coding: utf-8 -*-
"""
MWT collision graph manipulations
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import networkx as nx

def check_assumptions(graph):
    for node in graph:
        successors = graph.successors(node)
        predecessors = graph.predecessors(node)
        if len(successors) == 2:
            for successor, successor_indeg in graph.in_degree_iter(successors):
                if successor_indeg != 1:
                    print(node, '-->', successors)
                    raise AssertionError("Fission children have unexpected number of parents (not 1)")
        elif len(successors) == 1:
            successor = successors[0]
            successor_indeg = graph.in_degree(successor)
            if successor_indeg != 2:
                print(node, '-->', successors, 'indeg={}'.format(successor_indeg))
                raise AssertionError("Fusion child has unexpected number of parents (not 2)")

        if len(predecessors) == 2:
            for predecessor, predecessor_indeg in graph.out_degree_iter(predecessors):
                assert predecessor_indeg == 1, "Fission parents have unexpected number of children (not 1)"
        elif len(predecessors) == 1:
            assert graph.out_degree(predecessors[0]) == 2, "Fusion parent has unexpected number of children (not 2)"


def remove_chains(graph):
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

    The hidden data will be attached to the nodes in a simple list, for
    example from above: ``[A, (B, C), D]``.

    """
    all_nodes = graph.nodes()

    for node in all_nodes:
        pass#if node.wat
