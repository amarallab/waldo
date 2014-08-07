# -*- coding: utf-8 -*-
"""
Remove nodes that have no edges and no position data.
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import wio

__all__ = [
    'remove_blank_nodes',
]
def remove_blank_nodes(graph, experiment, nodes_with_data=None):
    all_nodes = set(graph.nodes(data=False))
    if nodes_with_data is None:
        if isinstance(experiment, wio.Experiment):
            term = experiment.prepdata.load('terminals').set_index('bid')
            nodes_with_data = list(term.index)
        else:
            raise ValueError('nodes_with_data must be provided if the'
                             'experiment is not a wio.Experiment.')

    suspected_blanks = all_nodes - set(nodes_with_data)
    blank_nodes = []
    for node in suspected_blanks:
        if list(graph.predecessors(node)):
            continue
        if list(graph.successors(node)):
            continue
        blank_nodes.append(node)
    for node in blank_nodes:
        graph.remove_node(node)


def remove_blank_nodes2(graph, experiment, nodes_with_data=None):
    """
    remove all nodes from graph that have absolutely no useful
    information (ie. no position and no edges to other nodes).


    params
    -----
    graph: (networkx graph)
       nodes are blob ids
    experiment: (wio.Experiment object)
       the experiment object corresponding to the same recording
    nodes_with_data: (list)
       nodes that have any xy data associated with them.

    """
    if nodes_with_data is None:
        if isinstance(experiment, wio.Experiment):
            term = experiment.prepdata.load('terminals').set_index('bid')
            nodes_with_data = list(term.index)
        else:
            raise ValueError('nodes_with_data must be provided if the'
                             'experiment is not a wio.Experiment.')
    blank_nodes = []
    for node in graph:
        if node in nodes_with_data:
            continue
        #print(node)

        children = list(graph.successors(node))
        if not parents and not children:
            blank_nodes.append(node)

    for node in blank_nodes:
        graph.remove_node(node)
