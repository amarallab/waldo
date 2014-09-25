# -*- coding: utf-8 -*-
"""
MWT collision graph manipulation general utilities
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import itertools
import collections
import types
import logging

# third party
import pandas as pd
import networkx as nx

# package specific
from waldo.extern import multiworm

__all__ = [
    'flat_node_list',
    'component_size_summary',
    #'suspected_collisions',
    'consolidate_node_data',
    'group_composite_nodes',
]

L = logging.getLogger(__name__)

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
            lifespan = graph.lifespan_f(node)
            if lifespan > threshold:
                L.debug('node {} was older than threshold ({} > {})'.format(node, lifespan, threshold))
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
        #print(node_data)
        if 'components' in node_data:
            internal_nodes = list(node_data['components'])
            internal_nodes.append(node)
        else:
            internal_nodes = [node]
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

def is_isolated(graph, node):
    """ returns True if node does not have any parents or children
    """
    return graph.degree(node) == 0

def consolidate_node_data(graph, experiment, node):
    """
    Returns a pandas DataFrame with all blob data for a node.
    Accepts both compound and single nodes.
    For compound nodes, data includes all components.

    params
    -----
    graph: (waldo.network.Graph object)
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
    data = []
    for subnode in graph.components(node):
        try:
            blob = experiment[subnode]
            df = blob.df
        except (KeyError, multiworm.MWTDataError) as e:
            print('{e} reading blob {i}'.format(e=e, i=subnode))
            continue

        if blob.empty:
            continue

        df.set_index('frame', inplace=True)
        df['blob'] = subnode
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

