# -*- coding: utf-8 -*-
"""
Resolving collisions using "unzip technique" pixel overlap
"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

import numpy as np
import pandas as pd
import random
import math

import encoding.decode_outline as de
from images.manipulations import points_to_aligned_matrix
from thinning.shape_thinning import skeletonize
from importing.skeletonize_outline import calculate_branch_and_endpoints

from ..util import consolidate_node_data

__all__ = [
    'find_potential_collisions',
    'unzip_resolve_collisions',
]

class CollisionException(Exception):
    pass


def find_potential_collisions(graph, min_duration, duration_factor):
    candidates = []
    for node in graph.nodes():
        succs = graph.successors(node)
        if len(succs) < 2:
            continue

        preds = graph.predecessors(node)
        if len(preds) < 2:
            continue

        min_succ_duration = None
        for s in succs:
            cur_preds = graph.predecessors(s)
            if len(cur_preds) != 1 or cur_preds[0] != node:
                min_succ_duration = None
                break
            duration = graph.node[s]['died'] - graph.node[s]['born']
            if min_succ_duration is None or min_succ_duration > duration:
                min_succ_duration = duration
        if min_succ_duration is None:
            continue

        min_pred_duration = None
        for p in preds:
            cur_succs = graph.successors(p)
            if len(cur_succs) != 1 or cur_succs[0] != node:
                min_pred_duration = None
                break
            duration = graph.node[p]['died'] - graph.node[p]['born']
            if min_pred_duration is None or min_pred_duration > duration:
                min_pred_duration = duration
        if min_pred_duration is None:
            continue

        duration = graph.node[node]['died'] - graph.node[node]['born']
        if duration >= min_duration and duration < min_succ_duration * duration_factor and duration < min_pred_duration * duration:
            candidates.append(node)
    for n in candidates:
        preds = (str(a) for a in graph.predecessors(n))
        succs = (str(a) for a in graph.successors(n))
        for a in graph.predecessors(n):
            succ = graph.successors(a)
            if len(succ) != 1 or succ[0] != n:
                print("ERROR AT %d: Pred: %s, Succ: %s" % (n, ", ".join(preds), ", ".join(succs)))
        for a in graph.successors(n):
            pred = graph.predecessors(a)
            if len(pred) != 1 or pred[0] != n:
                print("ERROR AT %d: Pred: %s, Succ: %s" % (n, ", ".join(preds), ", ".join(succs)))
    return candidates


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
            outline_points = de.decode_outline([x, y, l, enc])
            return outline_points
    else:
        print('Failed to find outline')
        print('grabbing', node, type(node))
        raise CollisionException

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
        raise CollisionException

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
        raise CollisionException
    return result


def calculate_distance_outline(outline):
    MAX_DIST = 1E100
    dist = MAX_DIST * (outline == 0)
    finish = False
    while not finish:
        finish = True
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                min_value = MAX_DIST
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if 0 <= i + x < dist.shape[0] and \
                                0 <= j + y < dist.shape[1]:
                            min_value = min(min_value, dist[i + x][j + y])
                if min_value + 1 < dist[i][j]:
                    dist[i][j] = min_value + 1
                    finish = False
    return dist


def unzip_resolve_collisions(graph, experiment, collision_nodes):
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
        preds = list(set(graph.predecessors(node)))
        if len(preds) != 2:
            continue

        succs = list(set(graph.successors(node)))
        if len(succs) != 2:
            continue

        a, b = preds  # 'up' and 'down' predecessors
        x, y = succs  # 'up' and 'down' successors

        # get the outlines
        outline_a = grab_outline(a, graph, experiment, first=False)
        outline_b = grab_outline(b, graph, experiment, first=False)
        outline_x = grab_outline(x, graph, experiment, first=True)
        outline_y = grab_outline(y, graph, experiment, first=True)
        outline_c_list = grab_outline_list(node, graph, experiment)

        # change to the same base
        outline_list = [outline_a, outline_b, outline_x, outline_y]
        outline_list.extend(outline_c_list)
        outline_list, bbox = points_to_aligned_matrix(outline_list)

        # get the outlines to the initial variables
        outline_a = outline_list[0]
        outline_b = outline_list[1]
        outline_x = outline_list[2]
        outline_y = outline_list[3]
        outline_c_list = outline_list[4:]

        MAX_STEPS = 50
        if len(outline_c_list) > MAX_STEPS:
            #step = int(len(outline_c_list) / MAX_STEPS)
            step = 2
            outline_c_list = outline_list[4::step]
        result = []

        count = 0
        while len(outline_c_list) > 0:
            current_c = outline_c_list.pop(0)

            area_removed_a = np.count_nonzero((outline_a - current_c) > 0)
            area_removed_b = np.count_nonzero((outline_b - current_c) > 0)

            new_a = outline_a * current_c
            new_b = outline_b * current_c

            dist_a = calculate_distance_outline(new_a)
            dist_b = calculate_distance_outline(new_b)

            skel_a = skeletonize(new_a)
            endpoints, _ = calculate_branch_and_endpoints(skel_a)
            endpoints_a = endpoints if len(endpoints) >= 2 else None

            skel_b = skeletonize(new_b)
            endpoints, _ = calculate_branch_and_endpoints(skel_b)
            endpoints_b = endpoints if len(endpoints) >= 2 else None

            # TODO: Use Nick's algorithm to create the approximation curve
            #       and use it adding a gaussian curve around it

            MIN_DIST = 5
            MAX_DIST = 10
            HUGE_DIST = 10
            remain = current_c - new_a - new_b
            possibles_a = []
            possibles_b = []
            for i in range(remain.shape[0]):
                for j in range(remain.shape[1]):
                    if remain[i][j] > 0:
                        da = dist_a[i][j]
                        db = dist_b[i][j]
                        if da < MIN_DIST and db > HUGE_DIST and area_removed_a > 0:
                            new_a[i][j] = 1
                            area_removed_a -= 1
                        if db < MIN_DIST and da > HUGE_DIST and area_removed_b > 0:
                            new_b[i][j] = 1
                            area_removed_b -= 1

                    if current_c[i][j] > 0:
                        if new_a[i][j] == 0 and dist_a[i][j] < MAX_DIST:
                            if len(endpoints_a) >= 2:
                                v1 = math.hypot(i - endpoints_a[0][0], j - endpoints_a[0][1])
                                v2 = math.hypot(i - endpoints_a[1][0], j - endpoints_a[1][1])
                                v = min(v1, v2)
                            else:
                                v = 5
                            possibles_a.append((i, j, dist_a[i][j] / v))

                        if new_b[i][j] == 0 and dist_b[i][j] < MAX_DIST:
                            if len(endpoints_b) >= 2:
                                v1 = math.hypot(i - endpoints_b[0][0], j - endpoints_b[0][1])
                                v2 = math.hypot(i - endpoints_b[1][0], j - endpoints_b[1][1])
                                v = min(v1, v2)
                            else:
                                v = 5
                            possibles_b.append((i, j, dist_b[i][j] / v))

            if area_removed_a > 0:
                for i, j, d in sorted(possibles_a, key=lambda x: x[2])[0:area_removed_a]:
                    new_a[i][j] = 1
                    area_removed_a -= 1

            if area_removed_b > 0:
                for i, j, d in sorted(possibles_b, key=lambda x: x[2])[0:area_removed_b]:
                    new_b[i][j] = 1
                    area_removed_b -= 1

            # skel_a = skeletonize(new_new_a)
            # endpoints, branchpoints = calculate_branch_and_endpoints(skel_a)
            # if len(branchpoints) == 0:
            #     new_a = new_new_a
            # else:
            #     print("NEW_A BRANCHPOINTS: %d" % len(branchpoints))
            #
            # skel_b = skeletonize(new_new_b)
            # endpoints, branchpoints = calculate_branch_and_endpoints(skel_b)
            # if len(branchpoints) == 0:
            #     new_b = new_new_b
            # else:
            #     print("NEW_B BRANCHPOINTS: %d" % len(branchpoints))

            common = new_a * new_b
            print("Removed areas: A %d, B %d" % (area_removed_a, area_removed_b))
            count -= 1
            if count <= 0:
                yield [outline_a, outline_b, current_c, new_a, new_b, common, outline_x, outline_y]
                count = 5

            outline_a = new_a
            outline_b = new_b


        cas = []      # 'up' intermediate nodes
        cbs = []      # 'down' intermediate nodes
        #return result

    #return collision_results