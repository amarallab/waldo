# -*- coding: utf-8 -*-
"""
Resolving collisions using "unzip technique" pixel overlap
"""
from __future__ import absolute_import, division, print_function
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import math
import itertools
import random

# third party
import numpy as np
import pandas as pd

# package specific
from waldo.images.manipulations import points_to_aligned_matrix
from waldo.thinning.shape_thinning import skeletonize
#from importing.skeletonize_outline import calculate_branch_and_endpoints
from .find_outlines import *
from .collision_overlap import compare_masks

from ..util import consolidate_node_data

__all__ = [
    'unzip_resolve_collisions',
]


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


def unzip_resolve_collisions(graph, experiment, collision_nodes, verbose=False, yield_bevahior=False):
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
            if verbose:
                print("Node %d, No preds: " % node, preds)
            continue

        succs = list(set(graph.successors(node)))
        if len(succs) != 2:
            if verbose:
                print("Node %d, No succs: " % node, succs)
            continue

        a, b = preds  # 'up' and 'down' predecessors
        x, y = succs  # 'up' and 'down' successors

        # get the outlines
        outline_a = grab_outline(a, graph, experiment, first=False)
        if outline_a is None:
            continue

        outline_b = grab_outline(b, graph, experiment, first=False)
        if outline_b is None:
            continue

        outline_x = grab_outline(x, graph, experiment, first=True)
        if outline_x is None:
            continue

        outline_y = grab_outline(y, graph, experiment, first=True)
        if outline_y is None:
            continue

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

        MAX_STEPS = 50
        if len(outline_c_list) > MAX_STEPS + 4:
            step = int((len(outline_c_list) - 4) / MAX_STEPS)
            outline_c_list = outline_list[4::step]
        else:
            outline_c_list = outline_list[4:]

        if yield_bevahior:
            yield [outline_a, outline_b, outline_x, outline_y]

        result = []
        while len(outline_c_list) > 0:
            result = compare_masks([(a, outline_a), (b, outline_b)], [(x, outline_x), (y, outline_y)])
            if len(result) > 0:
                if verbose:
                    print("%d, %d -> %d, %d, result: " % (a, b, x, y), result)
                graph.untangle_collision(node, result)
                break

            current_c = outline_c_list.pop(0)

            removed_a = outline_a - current_c
            removed_b = outline_b - current_c
            area_removed_a = np.count_nonzero(removed_a > 0)
            area_removed_b = np.count_nonzero(removed_b > 0)

            new_a = outline_a * current_c
            new_b = outline_b * current_c

            dist_a = calculate_distance_outline(new_a)
            dist_b = calculate_distance_outline(new_b)

            endpoints_a = []
            try:
                skel_a = skeletonize(new_a)
                endpoints, _ = calculate_branch_and_endpoints(skel_a)
                if endpoints is not None and len(endpoints) >= 2:
                    endpoints_a = endpoints
            except:
                pass

            endpoints_b = []
            try:
                skel_b = skeletonize(new_b)
                endpoints, _ = calculate_branch_and_endpoints(skel_b)
                if endpoints is not None and len(endpoints) >= 2:
                    endpoints_b = endpoints
            except:
                pass

            # Calculate percentage of the lose points close to endpoints
            if len(endpoints_a) == 0:
                percentage_adding_first_a = 0
            else:
                closed_points_to_endpoint = [0] * len(endpoints_a)
                for i, j in itertools.product(*[range(a) for a in removed_a.shape]):
                    if removed_a[i][j] > 0:
                        d = [math.hypot(i - ep[0], j - ep[1]) for ep in endpoints_a]
                        closed_points_to_endpoint[np.argmin(d)] += 1
                if closed_points_to_endpoint[0] == 0 and closed_points_to_endpoint[1] == 0:
                    percentage_adding_first_a = 0.5
                else:
                    percentage_adding_first_a = closed_points_to_endpoint[0] / sum(closed_points_to_endpoint)

            if len(endpoints_b) == 0:
                percentage_adding_first_b = 0
            else:
                closed_points_to_endpoint = [0] * len(endpoints_b)
                for i, j in itertools.product(*[range(a) for a in removed_b.shape]):
                    if removed_b[i][j] > 0:
                        d = [math.hypot(i - ep[0], j - ep[1]) for ep in endpoints_b]
                        closed_points_to_endpoint[np.argmin(d)] += 1
                if closed_points_to_endpoint[0] == 0 and closed_points_to_endpoint[1] == 0:
                    percentage_adding_first_b = 0.5
                else:
                    percentage_adding_first_b = closed_points_to_endpoint[0] / sum(closed_points_to_endpoint)


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
                        if da < MIN_DIST and db > HUGE_DIST:
                            new_a[i][j] = 1
                            area_removed_a -= 1
                        if db < MIN_DIST and da > HUGE_DIST:
                            new_b[i][j] = 1
                            area_removed_b -= 1

                    if current_c[i][j] > 0:
                        if new_a[i][j] == 0 and dist_a[i][j] < MAX_DIST:
                            if len(endpoints_a) >= 2:
                                v1 = math.hypot(i - endpoints_a[0][0], j - endpoints_a[0][1]) * percentage_adding_first_a
                                v2 = math.hypot(i - endpoints_a[1][0], j - endpoints_a[1][1]) * (1 - percentage_adding_first_a)
                                v = min(v1, v2)
                            else:
                                v = 5
                            possibles_a.append((i, j, dist_a[i][j] / v))

                        if new_b[i][j] == 0 and dist_b[i][j] < MAX_DIST:
                            if len(endpoints_b) >= 2:
                                v1 = math.hypot(i - endpoints_b[0][0], j - endpoints_b[0][1]) * percentage_adding_first_b
                                v2 = math.hypot(i - endpoints_b[1][0], j - endpoints_b[1][1]) * ( 1 - percentage_adding_first_b)
                                v = min(v1, v2)
                            else:
                                v = 5
                            possibles_b.append((i, j, dist_b[i][j] / v))

            while area_removed_a > 0:
                changed = False
                for i, j, d in sorted(possibles_a, key=lambda x: x[2])[0:area_removed_a]:
                    values = []
                    for ai, aj in itertools.product([-1, 0, 1], [-1, 0, 1]):
                        if 0 <= i + ai < new_a.shape[0] and 0 <= j + aj < new_a.shape[1]:
                            values.append(new_a[i + ai][j + aj])
                        else:
                            values.append(0)
                    valid = False
                    prev = values[-1]
                    for curr in values[0:-1]:
                        if curr == prev == 1:
                            valid = True
                            break
                        prev = curr

                    if valid:
                        new_a[i][j] = 1
                        area_removed_a -= 1
                        changed = True
                        break
                if not changed:
                    break

            while area_removed_b > 0:
                changed = False
                for i, j, d in sorted(possibles_b, key=lambda x: x[2])[0:area_removed_b]:
                    values = []
                    for ai, aj in itertools.product([-1, 0, 1], [-1, 0, 1]):
                        if 0 <= i + ai < new_b.shape[0] and 0 <= j + aj < new_b.shape[1]:
                            values.append(new_b[i + ai][j + aj])
                        else:
                            values.append(0)
                    valid = False
                    prev = values[-1]
                    for curr in values[0:-1]:
                        if curr == prev == 1:
                            valid = True
                            break
                        prev = curr

                    if valid:
                        new_b[i][j] = 1
                        area_removed_b -= 1
                        changed = True
                        break
                if not changed:
                    break

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
            if verbose:
                print("Removed areas: A %d, B %d" % (area_removed_a, area_removed_b))
            if yield_bevahior:
                yield [outline_a, outline_b, current_c, new_a, new_b, common, outline_x, outline_y]

            outline_a = new_a
            outline_b = new_b

        if len(result) == 0 and verbose:
            print("%d, %d -> %d, %d, NO result." % (a, b, x, y))
