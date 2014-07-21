# -*- coding: utf-8 -*-
"""
uses external information to remove bad nodes/chains.
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import pandas as pd

from ..util import is_isolated

#from annotation import image_validation as iv

__all__ = [
    #'add_validation',
    #'validate_nodes',
]

def add_validation(graph, ex_id):
    valid = iv.Validator(ex_id)
    good_list = valid.good_list()
    bad_list = valid.bad_list()

    for node in graph:
        nodes = [node]
        if type(node) == tuple:
            nodes = graph.node[node]['components']

        good = set([n for n in nodes if n in good_list])
        bad = set([n for n in nodes if n in bad_list])
        graph.node[node]['good'] = good
        graph.node[node]['bad'] = bad

def validate_nodes(graph, good_list, bad_list):
    """
    going to be the main function. not sure what it does yet.
    for now, just displays table.

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    good_list: (list)
       list of bids that are worms.
    bad_list: (list)
       list of bids that are not worms.
    """


    isolated_nodes, interacting_nodes = [], []
    for node in graph:
        if is_isolated(graph, node):
            isolated_nodes.append(node)
        else:
            interacting_nodes.append(node)

    isol = legitimacy_table(graph, nodes=isolated_nodes,
                            good_list=good_list, bad_list=bad_list)
    inter = legitimacy_table(graph, nodes=interacting_nodes,
                             good_list=good_list, bad_list=bad_list)

    print('\nisolated: {n}'.format(n=len(isolated_nodes)))
    print(isol)
    print('\ninteracting: {n}'.format(n=len(interacting_nodes)))
    print(inter)
    remove_isolated_bad(graph, good_list, bad_list)


def legitimacy_table(graph, nodes, good_list, bad_list=[]):
    """ creates a dataframe breaking down how many nodes
    have good and/or bad flags associated with them

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    nodes: (list)
       list of nodes to check
    good_list: (list)
       list of bids that are worms.
    bad_list: (list)
       list of bids that are not worms.

    returns
    ----
    l_table: (pandas DataFrame object)
        contains information about how many nodes fall into the
        following four catagories:
        good and bad, just good, just bad, neither.
    """
    data = [[0, 0], [0,0]]
    l_table = pd.DataFrame(data,
                          columns=['good', '-'],
                          index=['bad', '-'])

    for node in nodes:
        has_good, has_bad = check_node(graph, node, good_list, bad_list)
        hg = {True:'good', False:'-'}
        hb = {True:'bad', False:'-'}
        l_table.loc[hb[has_bad], hg[has_good]] += 1
    return l_table

def remove_isolated_bad(graph, good_list=[], bad_list=[]):
    """
    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    good_list: (list)
       list of bids that are worms.
    bad_list: (list)
       list of bids that are not worms.
    """
    bad_nodes = []
    for node in graph:
        if not is_isolated(graph, node):
            continue
        has_good, has_bad = check_node(graph, node, good_list, bad_list)
        if has_bad and not has_good:
            bad_nodes.append(node)

    for node in bad_nodes:
        graph.remove_node(node)


def check_node(graph, node, good_list, bad_list=[]):
    """ returns two boolean values specifying if node or any of it's
    components are in the good list or in the bad list.

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    node: (int or tuple)
       the name of the node to be checked
    good_list: (list)
       list of bids that are worms.
    bad_list: (list)
       list of bids that are not worms.
    """
    nodes = [node]
    if type(node) == tuple:
        nodes = graph.node[node]['components']
    # these values are true if any components in good or bad lists
    has_good = (len([n for n in nodes if n in good_list]) > 0)
    has_bad = (len([n for n in nodes if n in bad_list]) > 0)
    return has_good, has_bad
