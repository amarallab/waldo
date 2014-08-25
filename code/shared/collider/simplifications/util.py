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
import types

import pandas as pd
import networkx as nx

__all__ = [
    'flat_node_list',
    'component_size_summary',
    'suspected_collisions',
    'consolidate_node_data',
    'group_composite_nodes',
    'merge_bounds',
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

def flat_node_list(graph):
    """
    returns list of all non-compund nodes in a graph, inculding
    all the nodes that are inside compound nodes.
    """
    node_ids = []
    for node in graph:
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
    for n, G in enumerate(sorted(list(Gcc), key=len, reverse=True)[:10], start=1):
        print("{:>2d}. {:>5d} nodes : {}{}".format(
                n, len(G),
                ', '.join(str(node) for node, _ in zip(G.nodes_iter(), range(5))),
                '...' if len(G) > 5 else ''))

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

        node_life = digraph.lifespan(node)
        parents_life = [digraph.lifespan(p) for p in parents]
        children_life = [digraph.lifespan(c) for c in children]

        #if (sum(parents_life) + sum(children_life)) / (4 * node_life) > relative_threshold:
        if (sum(parents_life) / (2 * node_life) > relative_threshold and
                sum(children_life) / (2 * node_life) > relative_threshold):
            suspects.append(node)

    return suspects

def is_isolated(graph, node):
    """ returns True if node does not have any parents or children
    """
    return graph.degree(node) == 0


# def is_offshoot(graph, node, subnode):
#     """ returns True if subnode is offshoot.

#     params
#     ----
#     graph: (networkx graph object)
#        MUST BE NETWORK BEFORE COMPOUND NODES
#     node: (tuple)
#        the name of the compound node which the subnode is part of.

#     subnode: the name of the subnode we are testing.
#     """
#     if type(node) != tuple: #node not compound. not offshoot.
#         return False
#     elif subnode in node: #node is start or end. not offshoot.
#         return False
#     # since start/end gone.  no children or parents = offshoot
#     elif len(set(graph.successors(subnode))) == 0:
#         return True
#     elif len(set(graph.predecessors(subnode))) == 0:
#         return True
#     else: # has children and parents. not offshoot.
#         return False

#TODO: add remove offshoots?
def consolidate_node_data(graph, experiment, node):
    """
    Returns a pandas DataFrame with all blob data for a node.
    Accepts both compound and single nodes.
    For compound nodes, data includes all components.

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    experiment: (multiworm experiment object)
       the experiment from which data can be exctracted.
    node: (int or tuple)
       the id (from graph) for a node.

    returns
    -----
    all_data: (pandas DataFrame)
       index is 'frame'
       columns are:
       'area', 'centroid'
       'contour_encode_len', 'contour_encoded', 'contour_start',
       'midline', 'size', 'std_ortho', 'std_vector', 'time'
    """

    components = list(graph.node[node].get('components', [node]))
    #print('{n} components in {node}'.format(n=len(components),
    #                                        node=node))

    data = []
    for subnode in components:
        try:
            blob_data = dict(experiment[subnode])
        except Exception as e:
            print('{e} reading blob {i}'.format(e=e, i=subnode))
            continue
        if blob_data is None:
            continue
        if 'frame' not in blob_data:
            continue

        df = blob_data.df
        #df = pd.DataFrame(blob_data)
        df.set_index('frame', inplace=True)
        df['blob'] = subnode
        data.append(df)
        try:
            pass
        except Exception as e:
            print('df failed', blob_data)
            print(type(blob_data))
            for i in dir(blob_data):
                print(i)
            print(e)
            print(dict(blob_data))
            raise
        data.append(df)
    if data:
        all_data = pd.concat(data)
        all_data.sort(inplace=True)
        return all_data

# should find a better spot for this
def group_composite_nodes(experiment, graph, dataframe):
    cgen = ((node, graph.where_is(node)) for node in experiment.graph)
    cgen = (x for x in cgen if x[1] in graph)
    composites = pd.DataFrame(list(cgen), columns=['bid', 'composite_bid'])

    # merge, only keep ids in dataframe
    bounds = pd.merge(dataframe, composites, on='bid', how='left')

    # clean up and group
    bounds.drop('bid', axis=1, inplace=True)
    bounds.rename(columns={'composite_bid': 'bid'}, inplace=True)
    return bounds.groupby('bid')

# should find a better spot for this
def merge_bounds(experiment, graph):
    dataframe = experiment.prepdata.bounds
    gr = group_composite_nodes(experiment, graph, dataframe)
    return gr.agg({'x_min': min, 'x_max': max, 'y_min': min, 'y_max': max}).reset_index()
