__author__ = 'heltena'

import math
from importing.flags_and_breaks import fit_gaussian
import numpy as np

def dist_2d(c1, c2):
    xc = c1[0] - c2[0]
    yc = c1[1] - c2[1]
    return math.hypot(xc, yc)


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
            duration = graph.lifespan_f(s)
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
            duration = graph.lifespan_f(p)
            if min_pred_duration is None or min_pred_duration > duration:
                min_pred_duration = duration
        if min_pred_duration is None:
            continue

        duration = graph.lifespan_f(node)
        if duration >= min_duration and duration < min_succ_duration * duration_factor and duration < min_pred_duration * duration:
            candidates.append(node)
    return candidates


def find_area_based_collisions(graph, experiment, debug=False):
    ###TODO: Calculate Area Mean/Std correctly!!!
    terminals_df = experiment.prepdata.load('terminals')
    sizes_df = experiment.prepdata.load('sizes')
    moved_df = experiment.prepdata.load('moved')
    matches_df = experiment.prepdata.load('matches')

    matches_ids = list(set(matches_df[matches_df['good']]['bid']))
    #bl_thresh = 4
    #moving_ids = list(moved_df[moved_df['bl_moved'] > bl_thresh]['bid'])
    #print(moved_df.head())
    #print(len(moving_ids), 'moving ids')

    print(matches_df.head())
    print(len(matches_ids), 'matches ids')

    terminals_map = {int(v['bid']): i for i, v in terminals_df.iterrows()}
    sizes_map = {int(v['bid']): i for i, v in sizes_df.iterrows()}

    values = sizes_df['area_median'].loc[matches_ids]
    v = fit_gaussian(values, 50)
    if v is not None:
        mean, std = v
        print("I: Area mean gaussian: %f" % mean)
        print("I: Area std gaussian: %f" % std)

    print("I: Area mean before: %f" % sizes_df['area_median'].mean(axis=1))
    print("I: Area std before: %f" % sizes_df['area_median'].std(axis=1))
    area_mean = sizes_df['area_median'].loc[matches_ids].mean(axis=1)
    area_std = sizes_df['area_median'].loc[matches_ids].std(axis=1)

    #if debug:
    print("I: Area mean: %f, std: %f" % (area_mean, area_std))

    ###END

    def debug_data(x):
        terminals = terminals_df.iloc[terminals_map[x]]
        sizes = sizes_df.iloc[sizes_map[x]]
        pos0 = tuple(int(terminals[p]) for p in ['x0', 'y0', 't0'])
        posN = tuple(int(terminals[p]) for p in ['xN', 'yN', 'tN'])
        area = float(sizes['area_median'])
        return "%d (%s - %s) x %d" % (x, pos0, posN, area)

    candidates = []
    for node in graph.nodes():
        predecessors = graph.predecessors(node)
        if len(predecessors) != 2:
            continue

        successors = graph.successors(node)
        if len(successors) != 2:
            continue

        pred1, pred2 = predecessors
        succ1, succ2 = successors

        if node not in sizes_map \
                or pred1 not in sizes_map or pred2 not in sizes_map \
                or succ1 not in sizes_map or succ2 not in sizes_map:
            continue

        node_sizes = sizes_df.iloc[sizes_map[node]]
        pred1_sizes = sizes_df.iloc[sizes_map[pred1]]
        pred2_sizes = sizes_df.iloc[sizes_map[pred2]]
        succ1_sizes = sizes_df.iloc[sizes_map[succ1]]
        succ2_sizes = sizes_df.iloc[sizes_map[succ2]]

        node_area = float(node_sizes['area_median'])
        pred1_area = float(pred1_sizes['area_median'])
        pred2_area = float(pred2_sizes['area_median'])
        succ1_area = float(succ1_sizes['area_median'])
        succ2_area = float(succ2_sizes['area_median'])

        if not (area_mean - area_std <= node_area < area_mean + area_std):
            if debug:
                print("E: Bad 'node' area: (%d, %d) - %d - (%d, %d)" % (pred1, pred2, node, succ1, succ2))
                print("I: ", "\n  ".join(debug_data(x) for x in [pred1, pred2, node, succ1, succ2]))
            continue

        if pred1_area + pred2_area < area_mean + area_std:
            if debug:
                print("E: Bad 'pred1 + pred2' area: (%d, %d) - %d - (%d, %d)" % (pred1, pred2, node, succ1, succ2))
                print("I: ", "\n  ".join(debug_data(x) for x in [pred1, pred2, node, succ1, succ2]))
            continue

        if succ1_area + succ2_area < area_mean + area_std:
            if debug:
                print("E: Bad 'succ1 + succ2' area: (%d, %d) - %d - (%d, %d)" % (pred1, pred2, node, succ1, succ2))
                print("I: ", "\n  ".join(debug_data(x) for x in [pred1, pred2, node, succ1, succ2]))
            continue

        candidates.append((pred1, pred2, node, succ1, succ2))
    return candidates
