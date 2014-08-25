# -*- coding: utf-8 -*-
"""
Resolving collisions using pixel overlap
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

__all__ = [
    'resolve_collisions',
    'untangle_collision',
    'create_collision_masks',
    'compare_masks',
]

class CollisionException(Exception):
    pass

def grab_outline(node, graph, experiment, first=True):
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
            #print(node, x, y, l, enc)
            outline_points = de.decode_outline([x, y, l, enc])
            #print(outline_points)
            return outline_points
    else:
        print('Failed to find outline')
        print('grabbing', node, type(node))
        raise CollisionException

# TODO make this function agnostic to the number of parents.
def create_collision_masks(graph, experiment, node, verbose=False):
    """
    return lists of (node_id, binary mask) tuples for
    the parents and children of the specified node.

    binary masks are boolean np.arrays containing the filled area
    of the worm shape. all masks are in the same, reduced
    coordinate system.

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    experiment: (multiworm experiment object)
       the experiment object cooresponding to the same nodes
    node: (int or tuple)
       the id (in graph) used to identify a collision
    verbose: (bool)
       a toggle to turn on/off print statements

    returns
    -----
    parents: (list of tuples)
       each element in parents is a tuple of (node_id, mask_for_node)
    children: (list of tuples)
       same structure as parents.
    """
    p = list(set(graph.predecessors(node)))
    c = list(set(graph.successors(node)))
    if verbose:
        print('parents:{p}'.format(p=p))
        print('children:{c}'.format(c=c))
        #print('beginning:end pixel overlap')
    #grab relevant outlines.
    p0 = grab_outline(p[0], graph, experiment, first=False)
    p1 = grab_outline(p[1], graph, experiment, first=False)
    c0 = grab_outline(c[0], graph, experiment, first=True)
    c1 = grab_outline(c[1], graph, experiment, first=True)

    # align outline masks
    outline_list = [p0, p1, c0, c1]
    masks, bbox = points_to_aligned_matrix(outline_list)
    #reorganize
    parents = [i for i in zip(p, masks[:2])]
    children = [i for i in zip(c, masks[-2:])]
    return parents, children


#TODO: make agnostic to number of parents/children
def compare_masks(parents, children, err_margin=10, verbose=False):
    """
    returns a list of [parent, child] matches that maximize the
    amount of pixel overlap between parents and children.

    The winning matchup must have > err_margin pixels overlapping
    to be counted as a match. If this is not met, return empty list.

    For:
    parents:  A  B
               \/
               /\
    children  C  D

    the possible outcomes:
    (1) [[A, C], [B, D]]
    (2) [[A, D], [B, C]]
    (3) []
    if the number of overlapping pixes for (1) > (2) + err_margin,
    then (1) is the output.

    params
    -----
    parents: (list of tuples)
       Each element in parents is a (node_id, mask_for_node) tuple.
       This list must contain only two parents.
    children: (list of tuples)
       same structure as parents.
    err_margin: (int)
       number of pixels a match-up must exceed the other to be legit.
    verbose: (bool)
       a toggle to turn on/off print statements

    returns
    -----
    parent_child_matches: (list of lists)
       each element in the list contains a [parent, child] match as
       determined by the best set of overlapping
    """
    # TODO: remove assertation when number of comparisions irrelevant
    if len(parents) != 2 or len(children) != 2:
        raise ValueError('parents and children must be lists with two elements')

    d1_sum, d2_sum= 0, 0
    d1, d2 = [], []
    p_nodes, _ = zip(*parents)
    c_nodes, _ = zip(*children)
    comparison = pd.DataFrame(index=p_nodes, columns=c_nodes)
    for i, (p, p_mask) in enumerate(parents):
        for j, (c, c_mask) in enumerate(children):
            overlap = (p_mask & c_mask).sum()
            comparison.loc[p, c] = overlap
            if i == j:
                d1_sum += overlap
                d1.append([p, c])
            else:
                d2_sum += overlap
                d2.append([p, c])

    if verbose:
        print('prnts\tchildren')
        print(comparison)
        print('   diag \\   {s}   \t{d}'.format(s=d1_sum, d=d1))
        print('   diag /   {s}   \t{d}'.format(s=d2_sum, d=d2))

    if d1_sum > d2_sum + err_margin:
        return d1
    elif d2_sum > d1_sum + err_margin:
        return d2
    else:
        return []

def resolve_collisions(graph, experiment, collision_nodes):
    """

    Removes all the collisions that can be resolved
    through pixel-overlap from the graph.
    If a collision cannot be resolved, it remains in the graph.

    only works if collisions nodes have two parents and two children.

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    experiment: (multiworm experiment object)
       the experiment object cooresponding to the same nodes
    collision_nodes: (list)
       a list of nodes that are suspected to be collisions between
       worms.
    """
    collision_results = {}
    for node in collision_nodes:
        try:
            parent_masks, children_masks = create_collision_masks(graph, experiment, node)
            collision_result = compare_masks(parent_masks, children_masks)
        except CollisionException:
            print('Warning: {n} has insuficient parent/child data to resolve collision'.format(n=node))
            continue
        #print(node, 'is', collision_result)
        if collision_result:
            collision_results[node] = collision_result
            untangle_collision(graph, node, collision_result)


def untangle_collision(graph, collision_node, collision_result):
    """
    this untangles collisions by removing the collision
    nodes and merging the parent-child pairs that belong
    to the same worm.


    A   C
     \ /
   collision    ==>  (A, B) and (C, D)
     / \
    B   D

    The collision node and all nodes in collision result are
    removed from the graph and replaced by compound nodes.


    params
    --------
    graph: (networkx graph object)
    collision_node: (int or tuple)
       the node id (in graph) that identifies the collision.
    collision_result: (list)
       Values are lists of node pairs to be joined.

       example:
       [[node_A, node_B], [node_C, node_D]]
    """

    if collision_result:
        graph.remove_node(collision_node)

    #print(col)
    for (n1, n2) in collision_result:
        #print(cr)

        #parents = set(graph.predecessors(n1))
        #children = set(graph.successors(n2))

        # combine data
        #new_node, new_node_data = graph.condense_nodes(n1, n2)
        if 'collision' not in graph.node[n1]:
            graph.node[n1]['collisions'] = set()
        graph.node[n1]['collisions'].add(collision_node)

        # add merged node and link to previous parents/children
        #graph.add_node(new_node, **new_node_data)
        #graph.add_edges_from((p, new_node) for p in parents)
        #graph.add_edges_from((c, new_node) for c in children)
        graph.add_edge(n1, n2)
        # remove old nodes.
        #graph.remove_node(n1)
        #graph.remove_node(n2)
