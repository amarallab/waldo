# -*- coding: utf-8 -*-
"""
MWT collision graph manipulation general utilities
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import itertools
import collections

import networkx as nx

__all__ = [
    'flat_node_list',
    'lifespan',
    'component_size_summary',
    'suspected_collisions',
]

def _check_assumptions(graph):
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

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, six.string_types):
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

def lifespan(graph, node):
    return graph.node[node]['died'] - graph.node[node]['born']

def condense_nodes(graph, start, end, *others):
    """
    *start*, *end*, and *others* are all (key, value) pairs where the key is
    the node name and value is the node data in dictionary form.

    Required node dictionary keys:
      * born
      * died

    Optional node dictionary keys:
      * components
    """
    # come up with new node name
    if start != end:
        start_and_end = list(flatten([start, end]))
        new_node = start_and_end[0], start_and_end[-1]
    else:
        new_node = start

    # repackage data
    components = set()
    collisions = set()
    for node in itertools.chain([start, end], others):
        try:
            components.update(graph.node[node]['components'])
        except KeyError:
            components.add(node)
        try:
            collisions.update(graph.node[node]['collisions'])
        except KeyError:
            pass

    new_node_data = {
        'born': graph.node[start]['born'],
        'died': graph.node[end]['died'],
        'components': components,
        'collisions': collisions,
    }

    return new_node, new_node_data

def flat_node_list(graph):
    """
    returns list of all non-compund nodes in a graph, inculding
    all the nodes that are inside compound nodes.
    """
    node_ids = []
    for node in graph:
        if type(node) != tuple:
            node_ids.append(node)
        else:
            node_data = graph.node[node]
            if 'components' in node_data:
                internal_nodes = list(node)
            else:
                internal_nodes = list(node)
            node_ids.extend(internal_nodes)
    return list(set(node_ids))

def component_size_summary(graph):
    Gcc = nx.connected_component_subgraphs(graph.to_undirected())
    print("Component sizes and example nodes in descending order")
    for n, G in enumerate(Gcc[:10], start=1):
        print("{:>2d}. {:>5d} nodes : {}...".format(
                n, len(G), ', '.join([str(node) for node, _ in zip(G.nodes_iter(), range(5))])))

def suspected_collisions(digraph, relative_threshold):
    """
    From *digraph*, return a list of possible collision nodes.  The average
    duration of both parent and children nodes must exceed the suspected
    collision node times *relative_threshold*.
    """
    suspects = []
    for node in digraph:
        #print(node)
        parents = digraph.predecessors(node)
        children = digraph.successors(node)
        if len(parents) != 2 or len(children) != 2:
            continue

        node_life = collider.lifespan(digraph, node)
        parents_life = [collider.lifespan(digraph, p) for p in parents]
        children_life = [collider.lifespan(digraph, c) for c in children]

        #if (sum(parents_life) + sum(children_life)) / (4 * node_life) > relative_threshold:
        if (sum(parents_life) / (2 * node_life) > relative_threshold and
                sum(children_life) / (2 * node_life) > relative_threshold):
            suspects.append(node)

    return suspects
