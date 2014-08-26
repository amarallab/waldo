__author__ = 'heltena'

import math


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
    return candidates


def find_area_based_collisions(graph, experiment, debug=False):
    terminals_df = experiment.prepdata.load('terminals')
    sizes_df = experiment.prepdata.load('sizes')
    terminals_map = {int(v['bid']): i for i, v in terminals_df.iterrows()}
    sizes_map = {int(v['bid']): i for i, v in sizes_df.iterrows()}

    #TODO: Call flags_and_breaks.py fit_gaussian
    area_mean = sizes_df['area_median'].mean(axis=1)
    area_std = sizes_df['area_median'].std(axis=1)
    #area_mean, area_std =
    if debug:
        print("I: Area mean: %f, std: %f" % (area_mean, area_std))

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