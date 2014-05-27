# -*- coding: utf-8 -*-
"""
MWT collision graph manipulations
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools
import collections

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

def flatten(l, *no_recurse_types):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, six.string_types + no_recurse_types):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def frame_filter(threshold):
    def conditional(graph, nodes):
        for node in nodes:
            if graph.node[node]['died'] - graph.node[node]['born'] > threshold:
                return False
        return True
    return conditional

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

    The hidden data will be attached to the nodes as a set, for example from
    above: ``{A, B, C, D}``.

    """
    all_nodes = graph.nodes()

    while all_nodes:
        node = all_nodes.pop()
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

        try:
            cnode = (node, grandchild[-1])
        except TypeError:
            try:
                cnode = (node[0], grandchild)
            except TypeError:
                cnode = (node, grandchild)

        components = set()
        for n in flatten([node, children, grandchild], tuple):
            try:
                components.update(graph.node[n]['components'])
            except KeyError:
                components.add(n)

        cnode_data = {'components': components,
                      'born': graph.node[node]['born'],
                      'died': graph.node[grandchild]['died']}

        graph.add_node(cnode, **cnode_data)
        graph.add_edges_from((p, cnode) for p in parents)
        graph.add_edges_from((cnode, ggc) for ggc in greatgrandchildren)

        graph.remove_node(node)
        graph.remove_nodes_from(children)
        graph.remove_node(grandchild)

        all_nodes.append(cnode)

    #return graph

def family_tree(digraph, target):
    """
    Subsets graph to return predecessors and successors with a blood relation
    to the target node
    """
    all_successors = nx.bfs_tree(digraph, target, reverse=False)
    all_predecessors = nx.bfs_tree(digraph, target, reverse=True)
    subdig = digraph.subgraph(itertools.chain([target], all_successors, all_predecessors))
    return subdig

def nearby(digraph, target, max_distance):
    """
    Return a subgraph containing nodes within *max_distance* of *target* in
    *digraph*.
    """
    graph = digraph.to_undirected()
    lengths = nx.single_source_shortest_path_length(graph, target, max_distance + 1)
    nearby_nodes = set(node for node, length in six.iteritems(lengths) if length <= max_distance)
    nearby_plus = set(node for node, length in six.iteritems(lengths) if length == max_distance + 1)

    subdig = digraph.subgraph(nearby_nodes | nearby_plus).copy()
    for node in nearby_plus:
        subdig.node[node]['more'] = True

    return subdig
