# -*- coding: utf-8 -*-
"""
Subgraph methods for MWT collision graphs to show something managable, rather
than 10000+ nodes.
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools

import networkx as nx

def family_tree(digraph, target):
    """
    Subsets graph to return predecessors and successors with a blood relation
    to the target node
    """
    all_successors = nx.bfs_tree(digraph, target, reverse=False)
    all_predecessors = nx.bfs_tree(digraph, target, reverse=True)
    subdig = digraph.subgraph(itertools.chain(
            [target], all_successors, all_predecessors)).copy()
    return subdig

def nearby(digraph, target, max_distance):
    """
    Return a copy of a subgraph containing nodes within *max_distance* of
    *target* in *digraph*.
    """
    graph = digraph.to_undirected()
    lengths = nx.single_source_shortest_path_length(graph, target, max_distance + 1)
    nearby_nodes = set(node for node, length in six.iteritems(lengths)
                       if length <= max_distance)
    nearby_plus = set(node for node, length in six.iteritems(lengths)
                      if length == max_distance + 1)

    subdig = digraph.subgraph(nearby_nodes | nearby_plus).copy()
    for node in nearby_plus:
        subdig.node[node]['more'] = True

    return subdig
