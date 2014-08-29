# -*- coding: utf-8 -*-
"""
Helper functions related to outlines
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
import pandas as pd

import encoding.decode_outline as de
from images.manipulations import points_to_aligned_matrix

from ..util import consolidate_node_data

import matplotlib.pyplot as plt

__all__ = [
    'CollisionException',
    'grab_outline',
    'grab_outline_list',
]

class CollisionException(Exception):
    pass


def get_outlines(graph, experiment, nodes):
    outline_list = [grab_outline(node, graph, experiment, first=True) for node in nodes]
    outline_list, _ = points_to_aligned_matrix(outline_list)
    return outline_list


def grab_outline(node, graph, experiment, first=True, verbose=False):
    """
    return the first or last complete outline for a given node
    as a list of points.

    params
    -----
    node: (int or tuple)
       the id (from graph) for a node.
    graph: (networkx graph object)
       a directed graph of node interactions
    experiment: (multiworm experiment object)
       the experiment from which data can be exctracted.
    first: (bool)
       toggle that deterimines if first or last outline is returned

    returns
    ----
    outline: (list of tuples)
       a list of (x,y) points
    """

    nodes = [node]
    preds = graph.predecessors(node)
    while preds == 1:
        current = preds[0]
        nodes.insert(0, current)
        preds = graph.predecessors(current)

    if not first:
        nodes = nodes[::-1]

    node_count = len(nodes)
    while len(nodes) > 0:
        node = nodes.pop(0)
        df = consolidate_node_data(graph, experiment, node)
        if df is None:
            print('Failed to find node data')
            #print('grabbing', node, type(node))
            raise CollisionException
        if not first:
            df.sort(ascending=False, inplace=True)

        for frame, row in df.iterrows():
            x, y = row['contour_start']
            l = row['contour_encode_len']
            enc = row['contour_encoded']
            is_good = True
            if not enc or not l:
                is_good = False
            if not isinstance(enc, basestring):
                is_good = False
            if is_good:
                outline_points = de.decode_outline([x, y, l, enc])
                return outline_points
    if verbose:
        print('I: Failed to find outline in %d predeccessors' % node_count)
        print('I: grabbing', node, type(node))
    return None


def grab_outline_list(node, graph, experiment):
    """
    return the list of complete outline for a given node
    as a list of "list of points".

    params
    -----
    node: (int or tuple)
       the id (from graph) for a node.
    graph: (networkx graph object)
       a directed graph of node interactions
    experiment: (multiworm experiment object)
       the experiment from which data can be exctracted.

    returns
    ----
    outline_list: list of (list of tuples)
       a list of outlines, outline is a list of (x,y) points
    """

    df = consolidate_node_data(graph, experiment, node)
    if df is None:
        print('Failed to find node data')
        print('grabbing', node, type(node))
        return None

    result = []
    for frame, row in df.iterrows():
        x, y = row['contour_start']
        l = row['contour_encode_len']
        enc = row['contour_encoded']
        is_good = True
        if not enc or not l:
            is_good = False
        if not isinstance(enc, basestring):
            is_good = False
        if is_good:
            outline_points = de.decode_outline([x, y, l, enc])
            result.append(outline_points)
    if len(result) == 0:
        print('Failed to find outline')
        print('grabbing', node, type(node))
    return result