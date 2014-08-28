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

import matplotlib.pyplot as plt

__all__ = [
    'resolve_collisions',
    'untangle_collision',
    'create_collision_masks',
    'compare_masks',
]

class CollisionException(Exception):
    pass

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

# TODO make this function agnostic to the number of parents.
def create_collision_masks(graph, experiment, node, verbose=False,
                           report=False):
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
    report: (bool)
       returns a dict with information about how well function ran

    returns
    -----
    parents: (list of tuples)
       each element in parents is a tuple of (node_id, mask_for_node)
    children: (list of tuples)
       same structure as parents.

    [status report]: (dict)
       optional dictionary with values showing how well function ran.
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
    outline_list = [o for o in [p0, p1, c0, c1] if o is not None]
    #print(len(outline_list), 'outlines found for', node)
    if not outline_list:
        raise CollisionException
    masks, bbox = points_to_aligned_matrix(outline_list)

    next = 0

    if p0 is not None:
        p0 = masks[0]
        next += 1

    if p1 is not None:
        p1 = masks[next]
        next += 1

    if c0 is not None:
        c0 = masks[next]
        next += 1

    if c1 is not None:
        c1 = masks[next]

    parents = [(p[0], p0), (p[1], p1)]
    children = [(c[0], c0), (c[1], c1)]
    status_report = {'mask_count': len(outline_list)}
    if report:
        return parents, children, status_report
    else:
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

    NONE_OVERLAP_VALUE = 0
    d1_sum, d2_sum = 0, 0
    d1, d2 = [], []
    p_nodes, _ = zip(*parents)
    c_nodes, _ = zip(*children)
    comparison = pd.DataFrame(index=p_nodes, columns=c_nodes) if verbose else None
    for i, (p, p_mask) in enumerate(parents):
        for j, (c, c_mask) in enumerate(children):
            if p_mask is None or c_mask is None:
                overlap = NONE_OVERLAP_VALUE
            else:
                overlap = (p_mask & c_mask).sum()
            if verbose:
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
    result_report = {'resolved':[], 'missing_data':[], 'no_overlap':[]}
    for node in collision_nodes:
        try:
            a = create_collision_masks(graph, experiment, node,
                                       report=True)

            parent_masks, children_masks, report = a
            collision_result = compare_masks(parent_masks, children_masks)

            if report['mask_count'] < 4:
                result_report['missing_data'].append(node)

            #INFO: if you want to see the outmasks and the relation between them
            # data = [parent_masks[0], parent_masks[1], children_masks[0], children_masks[1]]
            # outmasks = {i: o for i, o in data if o is not None}
            #
            # v = [len(c) for c in collision_result]
            # if len(v) == 0:
            #     print("Problems in collision_result: ", collision_result, " node: ", node)
            # else:
            #     cols = max(v)
            #     f, axs = plt.subplots(cols, len(collision_result))
            #     for i, cr in enumerate(collision_result):
            #         for col, id in enumerate(cr):
            #             axs[i][col].axis('off')
            #             if id in outmasks:
            #                 axs[i][col].imshow(outmasks[id], interpolation='none')
            #     plt.show()

        except CollisionException:
            #print('Warning: {n} has insuficient parent/child data to resolve collision'.format(n=node))
            result_report['missing_data'].append(node)
            continue
        #print(node, 'is', collision_result)
        if collision_result:
            collision_results[node] = collision_result
            untangle_collision(graph, node, collision_result)
            result_report['resolved'].append(node)
        else:
            result_report['no_overlap'].append(node)
    return result_report

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
