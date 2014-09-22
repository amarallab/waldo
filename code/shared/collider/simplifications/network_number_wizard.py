__author__ = 'heltena'

import math
from collections import defaultdict
import numpy as np

def dist_Nd(a, b):
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(a, b)]))


def node_is_moving(node, terminals_df, MIN_DIST=10, MIN_ALIVE=0.25):
    try:
        row = terminals_df.loc[node]
        p0 = row['x0'], row['y0'], row['t0']
        p1 = row['xN'], row['yN'], row['tN']
        return dist_Nd(p0[0:2], p1[0:2]) > MIN_DIST and p1[2] - p0[2] > MIN_ALIVE
    except KeyError as e:
        return False


def area_for_node(node, terminals_df, sizes_df, default=None, MIN_ALIVE=0.25):
    try:
        t0 = terminals_df.loc[node]['t0']
        tN = terminals_df.loc[node]['tN']
        alive = float(tN) - float(t0)
        if alive < MIN_ALIVE:
            return default
        value = sizes_df.loc[node]['area_median']
        if math.isnan(value):
            return default
        return value
    except KeyError as e:
        return default


def round_worm_count_value(v):
    return int(v / 0.5) * 0.5


def network_number_wizard(graph, experiment, verbose=False):
    sizes_df = experiment.prepdata.load('sizes')
    terminals_df = experiment.prepdata.load('terminals')
    sizes_df.set_index('bid', inplace=True)
    terminals_df.set_index('bid', inplace=True)

    # assign '1'
    areas = []
    worm_count = defaultdict(float)
    seeds = set()
    one_assigned = set()
    for node in graph.nodes():
        if node_is_moving(node, terminals_df):
            area = area_for_node(node, terminals_df, sizes_df)
            if area is not None:
                areas.append(area)
            worm_count[node] = 1
            seeds.add(node)
            one_assigned.add(node)
    mean, std = np.mean(areas), np.std(areas)
    if verbose:
        print("I: Mean: {mean}, std: {std}, l={l}".format(mean=mean, std=std, l=len(areas)))

    print("one_assigned count: {l}".format(l=len(one_assigned)))
    # assign successor values
    pred_count = defaultdict(int)
    while len(seeds) > 0:
        current = seeds.pop()
        wc = worm_count[current]

        # calculate percentage for all of the successor list
        succ_area = sum([area_for_node(succ, terminals_df, sizes_df, 0)
                         for succ in graph.successors(current)
                         if succ is not one_assigned])

        # Add percentage for each successor
        for succ in graph.successors(current):
            if succ in one_assigned:
                continue
            if succ_area == 0:
                value = wc / len(graph.successors(current))
            else:
                value = wc * area_for_node(succ, terminals_df, sizes_df, 0) / succ_area

            worm_count[succ] += value
            pred_count[succ] += 1

            if pred_count[succ] == len(graph.predecessors(succ)):
                seeds.add(succ)
                del(pred_count[succ])

    for k in pred_count:
        del(worm_count[k])

    # assign area based value for unassigned ones
    for node in graph.nodes():
        if node not in worm_count or worm_count[node] == 0:
            area = area_for_node(node, terminals_df, sizes_df, None, 0)
            if area is not None:
                count = 0.5
                max_value = mean - std / 2.
                while area > max_value:
                    count += 0.5
                    max_value += std / 2.
                worm_count[node] = count
            else:
                worm_count[node] = 1
    for node in worm_count:
        worm_count[node] = round_worm_count_value(worm_count[node])

    return worm_count
