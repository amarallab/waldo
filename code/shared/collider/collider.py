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


def remove_chains(graph, conditional=None):
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
        if node not in graph:
            continue # node was already removed/abridged

        parents = set(graph.predecessors(node))
        children = set(graph.successors(node))
        grandchildren = set()
        for child in children:
            grandchildren.update(graph.successors_iter(child))

        if len(grandchildren) != 1:
            continue # skip doing anything

        if conditional is not None:
            if conditional(graph, children):
                pass
            else:
                continue # children fail conditional testing

        grandchild = grandchildren.pop()
        greatgrandchildren = set(graph.successors(grandchild))

        cnode = '{}-{}'.format(node, grandchild)
        cnode_data = {'components': [node, children, grandchild]}

        graph.add_node(cnode, **cnode_data)
        graph.add_edges_from((p, cnode) for p in parents)
        graph.add_edges_from((cnode, ggc) for ggc in greatgrandchildren)

        graph.remove_node(node)
        graph.remove_nodes_from(children)
        graph.remove_node(grandchild)

    #return graph



