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
import itertools
from .find_outlines import *

from ..util import consolidate_node_data

import matplotlib.pyplot as plt

__all__ = [
    'create_collision_masks',
    'compare_masks',
    'generalized_compare_masks',
    'resolve_collisions',
    'resolve_multicollisions',
    'untangle_collision',
]

# for 2 parents, 2 children
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

# for X parents, Y children (X, Y can be anything)
def generalized_create_collision_masks(graph, experiment, parents, children, verbose=False,
                           report=False):
    """
    return lists of (node_id, binary mask) tuples for
    lists of parent and children nodes.

    binary masks are boolean np.arrays containing the filled area
    of the worm shape. all masks are in the same, reduced
    coordinate system.

    params
    -----
    graph: (networkx graph object)
       a directed graph of node interactions
    experiment: (multiworm experiment object)
       the experiment object cooresponding to the same nodes
    parents: (list)
       the list of node ids (in graph) used to identify parents of a collision
    children: (list)
       the list of node ids (in graph) used to identify children of a collision
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

    if verbose:
        print('parents:{p}'.format(p=parents))
        print('children:{c}'.format(c=children))


    # grab outline data
    outlines = {}
    not_none = []
    for p in parents:
        o = grab_outline(p[0], graph, experiment, first=False)
        outlines[p] = o
        if o:
            not_none.append(p)
    for c in children:
        o = grab_outline(c[0], graph, experiment, first=True)
        outlines[c] = o
        if o:
            not_none.append(p)

    outline_list = [outlines[i] for i in not_none]
    if not non_none:
        raise CollisionException

    # align outline masks
    mask_dict = {}
    masks, bbox = points_to_aligned_matrix(outline_list)
    for node, mask in zip(not_none, masks):
        mask_dict[node] = mask

    parent_data = [(p, mask_dict.get(p, None)) for p in parents]
    child_data = [(c, mask_dict.get(c, None)) for c in children]

    status_report = {'parent_count': len(parents),
                     'child_count': len(children),
                     'mask_count': len(non_none)}
    if report:
        return parent_data, child_data, status_report
    else:
        return parent_data, child_data

def generalized_compare_masks(parents, children, err_margin=10, verbose=False):
    """
    returns a list of [parent, child] matches that maximize the
    amount of pixel overlap between parents and children.

    The winning matchup must have > err_margin pixels overlapping
    to be counted as a match. If this is not met, return empty list.

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
    none_overlap_value = 0
    p_nodes, _ = zip(*parents)
    c_nodes, _ = zip(*children)
    comparison = pd.DataFrame(index=p_nodes, columns=c_nodes)
    #if verbose else None
    for i, (p, p_mask) in enumerate(parents):
        for j, (c, c_mask) in enumerate(children):
            if p_mask is None or c_mask is None:
                overlap = none_overlap_value
            else:
                overlap = (p_mask & c_mask).sum()
            comparison.loc[p, c] = overlap
    print(comparison)

    def df_to_sol(df, reformat_str=True):
        """
        converts dataframe columns and index to solution lists


        """
        p_nodes = list(df.index)
        c_nodes = list(df.columns)
        #best_solution = list(zip(p_nodes, c_shuffle[rank[0]]))

        def reformat_combined_entries(nodes):
            nodes2 = []
            for n in nodes:
                if type(n) == str:
                    n = [int(i) for i in n.split('-')]
                nodes2.append(n)
            return nodes2

        if reformat_str:
            p_nodes = reformat_combined_entries(p_nodes)
            c_nodes = reformat_combined_entries(c_nodes)

        return list(zip(p_nodes, c_nodes))

    def parent_child_match(comparison):
        """
        gets best solution when there are equal numbers of parents
        and children
        """
        scores = []
        n_worms = len(comparison)
        c_nodes = comparison.columns
        c_shuffle = list(itertools.permutations(c_nodes, n_worms))
        i = np.identity(n_worms)
        #print(c_shuffle)
        dfs = []
        for c_order in c_shuffle:
            df = comparison[list(c_order)] * i
            scores.append(df.sum().sum())
            dfs.append(df)

        scores = np.array(scores)
        rank = np.argsort(scores)[::-1]
        best_score = scores[rank[0]]
        second_best = scores[rank[1]]

        #best_solution = list(zip(p_nodes, c_shuffle[rank[0]]))
        best_df = dfs[rank[0]]
        return best_score, second_best, best_df

    def parent_child_mismatch(comparison):
        """
        gets best solution when there is a mismatch between numbers
        of parents and numbers of children.

        """
        h, w = comparison.shape
        transpose = h > w
        if transpose:
            comparison = comparison.T
        # TODO:
        # 1. create comparisons
        best_scores = []
        second_best_scores = []
        dfs = []

        c_nodes = comparison.columns
        n_worms = len(c_nodes)
        #print(c_nodes)

        c_shuffle = list(itertools.permutations(c_nodes, n_worms))
        #print(c_shuffle)
        for c in c_shuffle:
            #print('hey', c)
            new_key = str('{a}-{b}'.format(a=c[0], b=c[1]))
            #print('new key', new_key)

            df = pd.DataFrame(index=comparison.index)
            df[new_key] = comparison[c[0]] + comparison[c[1]]
            df[c[-1]] = comparison[c[-1]]
            #print(df)

            if len(df.index) == len(df.columns):
                bs, sb, best_df = parent_child_match(df)
            else:
                bs, sb, best_df = parent_child_mismatch(df)
            best_scores.append(bs)
            second_best_scores.append(sb)
            dfs.append(best_df)

        scores = np.array(best_scores)
        rank = np.argsort(scores)[::-1]
        #print(scores)
        #print(rank)

        best_score= scores[rank[0]]
        best_df = dfs[rank[0]]
        if transpose:
            best_df = best_df.T

        # second best
        second_best = max([best_scores[1], max(second_best_scores)])
        return best_score, second_best, best_df

    # if simple solution if parents and children in same
    if len(p_nodes) == len(c_nodes):
        print('parent child match')
        best_score, second_best, best_df = parent_child_match(comparison)

    else:
        print('parent child mismatch')
        best_score, second_best, best_df = parent_child_mismatch(comparison)

    print(best_df)
    best_solution = df_to_sol(best_df)
    if best_score < err_margin:
        return [], 'no-overlap'
    elif best_score < second_best + err_margin:
        return [], 'close'
    else:
        return best_solution, 'success'


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
    result_report = {'resolved': [], 'missing_data': [], 'no_overlap': [],
                     'collision_lifespans_t': {}}
    for node in collision_nodes:
        try:
            a = create_collision_masks(graph, experiment, node,
                                       report=True)

            parent_masks, children_masks, report = a
            collision_result = compare_masks(parent_masks, children_masks)

            ####### UNCOMMENT to start resolving MULTI-COLLISIONS
            # a = generalized_create_collision_masks(graph, experiment, node,
            #                            report=True)
            # parent_masks, children_masks, report = a
            # collision_result = generalized_compare_masks(parent_masks, children_masks)
            ####################################################

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
            result_report['collision_lifespans_t'][node] = graph.lifespan_t(node)
            collision_results[node] = collision_result
            untangle_collision(graph, node, collision_result)
            result_report['resolved'].append(node)

            # temporary reporting to track down where long tracks dissapear to.
            long_collisions = [l for l in result_report['collision_lifespans_t'].values() if l > 60 * 20]
            if long_collisions:
                print(len(long_collisions), 'collisions removed that were longer than 20min')
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


# WORKING COPY TO GET MULI-COLLISIONS WORKING
def untangle_collision2(graph, collision_node, collision_result):
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

    for (n1, n2) in collision_result:
        merge_types = [list, tuple]

        # make single value lists as a default for parent-child matches
        p_nodes = [n1]
        c_nodes = [n2]

        # use n1 or n2 if either is already is a list of nodes
        if n1 in merge_types:
            p_nodes = n1
        if n2 in merge_types:
            c_nodes = n2

        # for each parent and each child, draw one link.
        for p in p_nodes:
            for c in c_nodes:
                graph.add_edge(p, c)

            # add record of collision to the parent node
            if 'collision' not in graph.node[p]:
                graph.node[p]['collisions'] = set()
            graph.node[p]['collisions'].add(collision_node)



def resolve_multicollisions(graph, experiment, collision_nodes):
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
    result_report = {'resolved': [], 'missing_data': [], 'no_overlap': [],
                     'collision_lifespans_t': {}}
    for node in collision_nodes:
        try:
            a = generalized_create_collision_masks(graph, experiment, node,
                                                   report=True)
            parent_masks, children_masks, report = a
            collision_result = generalized_compare_masks(parent_masks, children_masks)

            # TODO make this more general
            if report['mask_count'] < 4:
                result_report['missing_data'].append(node)

        except CollisionException:
            #print('Warning: {n} has insuficient parent/child data to resolve collision'.format(n=node))
            result_report['missing_data'].append(node)
            continue
        #print(node, 'is', collision_result)
        if collision_result:
            result_report['collision_lifespans_t'][node] = graph.lifespan_t(node)
            collision_results[node] = collision_result
            # this is for multi
            untangle_collision2(graph, node, collision_result)
            result_report['resolved'].append(node)

            # temporary reporting to track down where long tracks dissapear to.
            long_collisions = [l for l in result_report['collision_lifespans_t'].values() if l > 60 * 20]
            if long_collisions:
                print(len(long_collisions), 'collisions removed that were longer than 20min')
        else:
            result_report['no_overlap'].append(node)
    return result_report
