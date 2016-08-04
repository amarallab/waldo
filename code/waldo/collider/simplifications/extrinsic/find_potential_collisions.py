from __future__ import absolute_import, print_function
from six.moves import range

__author__ = 'heltena'

# standard library
import math

# third party
import numpy as np
import scipy

# package specific

__all__ = [
    'find_bbox_based_collisions',
    'find_time_based_collisions',
    'find_area_based_collisions',
]

# def dist_2d(c1, c2):
#     xc = c1[0] - c2[0]
#     yc = c1[1] - c2[1]
#     return math.hypot(xc, yc)

def fit_gaussian(x, num_bins=200):
    # some testdata has no variance whatsoever, this is escape clause
    if math.fabs(max(x) - min(x)) < 1e-5:
        print('fit_gaussian exit')
        return max(x), 1

    n, bin_edges = np.histogram(x, num_bins, normed=True)
    bincenters = [0.5 * (bin_edges[i + 1] + bin_edges[i]) for i in range(len(n))]

    # Target function
    #fitfunc = lambda p, x: mlab.normpdf(x, p[0], p[1])
    fitfunc = lambda p, x: scipy.stats.norm.pdf(x, p[0], p[1])
    # Distance to the target function
    errfunc = lambda p, x, y: fitfunc(p, x) - y

    # Initial guess for the parameters
    mu = np.mean(x)
    sigma = np.std(x)
    p0 = [mu, sigma]
    p1, success = scipy.optimize.leastsq(errfunc, p0[:], args=(bincenters, n))
    # weirdly if success is an integer from 1 to 4, it worked.
    if success in [1,2,3,4]:
        mu, sigma = p1
        return mu, sigma
    else:
        return None

def _iterate_collisions_by_structure(graph, num_succ_range=[2, 3], num_pred_range=[2, 3]):
    for node in graph.nodes():
        succs = graph.successors(node)
        if not (num_succ_range[0] <= len(succs) < num_succ_range[1]):
            continue

        preds = graph.predecessors(node)
        if not (num_pred_range[0] <= len(preds) < num_pred_range[1]):
            continue

        yield (node, preds, succs)


def find_time_based_collisions(graph, min_duration, duration_factor):
    candidates = []
    for node, preds, succs in _iterate_collisions_by_structure(graph, [1, 2000], [1, 2000]):
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


def find_bbox_based_collisions(graph, experiment, min_distance=10, verbose=False):
    bounds_df = experiment.prepdata.load('bounds')
    bounds_df.set_index('bid', inplace=True)
    candidates = []
    for node, preds, _ in _iterate_collisions_by_structure(graph):
        moved = False
        for pred in preds:
            if pred in bounds_df.index:
                try:
                    bdata = bounds_df.loc[pred]
                except:
                    continue
                x_min, x_max, y_min, y_max = bdata['x_min'], bdata['x_max'], bdata['y_min'], bdata['y_max']
                distance = (x_max - x_min) ** 2 + (y_max - y_min) ** 2
                if distance > min_distance:
                    moved = True
                    break
        if moved:
            candidates.append(node)
    return candidates


def find_area_based_collisions(graph, experiment, verbose=False):
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

    #print(matches_df.head())
    #print(len(matches_ids), 'matches ids')

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

    #if verbose:
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
    for node, preds, succs in _iterate_collisions_by_structure(graph):
        pred1, pred2 = preds
        succ1, succ2 = succs

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
            if verbose:
                print("E: Bad 'node' area: (%d, %d) - %d - (%d, %d)" % (pred1, pred2, node, succ1, succ2))
                print("I: ", "\n  ".join(debug_data(x) for x in [pred1, pred2, node, succ1, succ2]))
            continue

        if pred1_area + pred2_area < area_mean + area_std:
            if verbose:
                print("E: Bad 'pred1 + pred2' area: (%d, %d) - %d - (%d, %d)" % (pred1, pred2, node, succ1, succ2))
                print("I: ", "\n  ".join(debug_data(x) for x in [pred1, pred2, node, succ1, succ2]))
            continue

        if succ1_area + succ2_area < area_mean + area_std:
            if verbose:
                print("E: Bad 'succ1 + succ2' area: (%d, %d) - %d - (%d, %d)" % (pred1, pred2, node, succ1, succ2))
                print("I: ", "\n  ".join(debug_data(x) for x in [pred1, pred2, node, succ1, succ2]))
            continue

        candidates.append(node)
    return candidates
