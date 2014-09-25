from __future__ import absolute_import, print_function

__author__ = 'heltena'

import math

__all__ = [
    'find_potential_cut_worms',
]

def dist_2d(c1, c2):
    xc = c1[0] - c2[0]
    yc = c1[1] - c2[1]
    return math.hypot(xc, yc)


# max_first_last_distance is the max distance between the first node and the last node (not between siblings)
# max_sibling_distance is the max distance between the first node and each of its sibling
def find_potential_cut_worms(graph, experiment, max_first_last_distance=40, max_sibling_distance=50, debug=False):
    terminals_df = experiment.prepdata.load('terminals')
    sizes_df = experiment.prepdata.load('sizes')
    terminals_map = {int(v['bid']): i for i, v in terminals_df.iterrows()}
    sizes_map = {int(v['bid']): i for i, v in sizes_df.iterrows()}

    area_mean = sizes_df['area_median'].mean(axis=1)
    area_std = sizes_df['area_median'].std(axis=1)
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
        successors = graph.successors(node)

        if len(successors) != 2:
            continue

        sibling1, sibling2 = successors
        if len(graph.predecessors(sibling1)) != 1 or len(graph.predecessors(sibling2)) != 1:
            continue

        sibling1_successors = graph.successors(sibling1)
        sibling2_successors = graph.successors(sibling2)
        if len(sibling1_successors) != 1 or sibling1_successors != sibling2_successors:
            continue

        last = sibling1_successors[0]
        if last not in terminals_map or sibling1 not in terminals_map or sibling2 not in terminals_map:
            if debug:
                print("E: Problems with nodes: %d - (%d, %d) - %d" % (node, sibling1, sibling2, last))
            continue

        node_terminals = terminals_df.iloc[terminals_map[node]]
        last_terminals = terminals_df.iloc[terminals_map[last]]
        node_pos = tuple(float(node_terminals[p]) for p in ['xN', 'yN', 'tN'])
        last_pos = tuple(float(last_terminals[p]) for p in ['x0', 'y0', 't0'])
        if dist_2d(node_pos, last_pos) >= max_first_last_distance:
            if debug:
                print("I: Distance between 'node' and 'last' is greater than %f: %d - (%d, %d) - %d" % \
                      (max_first_last_distance, node, sibling1, sibling2, last))
                print(" ", "\n  ".join(debug_data(x) for x in [node, sibling1, sibling2, last]))
            continue

        sibling1_terminals = terminals_df.iloc[terminals_map[sibling1]]
        sibling2_terminals = terminals_df.iloc[terminals_map[sibling2]]
        sibling1_pos = tuple(float(sibling1_terminals[p]) for p in ['x0', 'y0', 't0'])
        sibling2_pos = tuple(float(sibling2_terminals[p]) for p in ['x0', 'y0', 't0'])
        if max(dist_2d(node_pos, sibling1_pos), dist_2d(node_pos, sibling2_pos)) >= max_sibling_distance:
            if debug:
                print("I: Distance between siblings is greater than %f: %d - (%d, %d) - %d" % \
                      (max_sibling_distance, node, sibling1, sibling2, last))
                print(" ", "\n  ".join(debug_data(x) for x in [node, sibling1, sibling2, last]))
            continue

        node_sizes = sizes_df.iloc[sizes_map[node]]
        last_sizes = sizes_df.iloc[sizes_map[last]]
        sibling1_sizes = sizes_df.iloc[sizes_map[sibling1]]
        sibling2_sizes = sizes_df.iloc[sizes_map[sibling2]]

        node_area = float(node_sizes['area_median'])
        last_area = float(last_sizes['area_median'])
        sibling1_area = float(sibling1_sizes['area_median'])
        sibling2_area = float(sibling2_sizes['area_median'])

        if not (area_mean - area_std <= node_area < area_mean + area_std):
            if debug:
                print("E: Bad 'node' area: %d - (%d, %d) - %d" % (node, sibling1, sibling2, last))
                print("I: ", "\n  ".join(debug_data(x) for x in [node, sibling1, sibling2, last]))
            continue

        if not (area_mean - area_std <= last_area < area_mean + area_std):
            if debug:
                print("E: Bad 'last' area: %d - (%d, %d) - %d" % (node, sibling1, sibling2, last))
                print("I: ", "\n  ".join(debug_data(x) for x in [node, sibling1, sibling2, last]))
            continue

        if not ((area_mean - area_std) / 2 <= sibling1_area < (area_mean + area_std) / 2) or \
                not ((area_mean - area_std) / 2 <= sibling2_area < (area_mean + area_std) / 2):
            if debug:
                print("E: Sibling area problems: %d - (%d, %d) - %d" % (node, sibling1, sibling2, last))
                print("I: ", "\n  ".join(debug_data(x) for x in [node, sibling1, sibling2, last]))
            continue

        candidates.append([node, sibling1, sibling2, last])

    result = []
    while len(candidates) > 0:
        current = candidates.pop(0)
        used = False
        for i in range(len(result)):
            a = result[i]
            if a[-1] == current[0]:
                a.extend(current[1:])
                used = True
                break
            elif a[0] == current[-1]:
                current.extend(a[1:])
                result[i] = current
                used = True
                break
        if not used:
            result.append(current)

    return result
