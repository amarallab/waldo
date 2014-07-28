#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library

# third party

# project specific
from .scorer import Scorer

# parameters
MAX_CELERITY = 5  # pixel / seconds
MAX_TIME = 5
OFFSET = 10 # pixels added to the radius of the cone

def dist_2d(c1, c2):
    xc = c1[0] - c2[0]
    yc = c1[1] - c2[1]
    return math.hypot(xc, yc)


def dist_3d(c1, c2):
    d = [c1[i]-c2[i] for i in range(3)]
    return math.sqrt(sum([x * x for x in d]))


def angle_2d(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx) % (2 * math.pi)


# class DataRow:
#     def __init__(self, id, t0, tN, x0, xN, y0, yN):
#         self.id = id
#         self.begin = (x0, y0, t0)
#         self.end = (xN, yN, tN)
#     def __repr__(self):
#         return "(%d, [%f:%f,%f]->[%f:%f,%f])" % tuple([self.id] + list(self.begin) + list(self.end))


class Taper(object):
    """
    Designed to take a wio.Experiment-like object.
    """
    def __init__(self, experiment, regenerate_cache=False):
        self.experiment = experiment

        self.scorer = Scorer(experiment)





# dl is an array of DataRow ordered by "begin time"
def index_of_data_with_begin_time_greater_than(dl, t):
    first = 0
    last = len(dl)
    while abs(first - last) > 1:
        middle = (first + last) / 2
        middle_value = dl[middle].begin[2]
        if middle_value < t:
            first = middle
        else:
            last = middle
    while first < len(dl) and dl[first].begin[2] <= t:
        first += 1
    return first



def inside_time_cone(tail, head, max_celerity, max_time, offset):
    if tail.end == head.begin:
        return True
    dt = head.begin[2] - tail.end[2]
    if dt < 0 or dt >= max_time:
        return False
    radius = offset + max_celerity * dt
    return dist_2d(tail.end, head.begin) < radius


def calculate_score(tail, head):
    dist = dist_3d(tail.end, head.begin)
    if dist == 0:
        return 10000.0
    else:
        return 1.0/dist


def calculate_relations(network, terminals):
    """
    params
    -----
    network: (networkx graph object)
    terminals: (dataframe)
    """

    #basedir, prefix):

    # READ ALL THE TERMINALS
    # filename = "{basedir}/prep/{prefix}/{prefix}-terminals.csv".format(basedir=basedir, prefix=prefix)
    # with open(filename, "rt") as f:
    #     data_list = []
    #     data_map = {}
    #     minT = None
    #     maxT = None
    #     for line in f.readlines()[1:]:
    #         values = line.strip().split(',')
    #         bid = int(values[0])
    #         params = [bid] + [float(x) for x in values[1:]]
    #         data = DataRow(*params)
    #         data_list.append(data)
    #         data_map[data.bid] = data
    #         if minT is None:
    #             minT = min(data.begin[2], data.end[2])
    #         else:
    #             minT = min(minT, data.begin[2], data.end[2])
    #         if maxT is None:
    #             maxT = max(data.begin[2], data.end[2])
    #         else:
    #             maxT = max(maxT, data.begin[2], data.end[2])

    # data_list = sorted(data_list, key=lambda x: x.begin[2]) # sorting by begin time
    print "Total nodes readed: ", len(terminals)

    data_index_nodes = set(terminals['bid'])
    in_index_nodes = set([x for x in network.nodes() if len(network.predecessors(x)) == 0]) & data_index_nodes
    out_index_nodes = set([x for x in network.nodes() if len(network.successors(x)) == 0]) & data_index_nodes

    # TODO: resume recoding here.
    in_nodes = sorted([data_map[x] for x in in_index_nodes], key=lambda x: x.begin[2])
    out_nodes = sorted([data_map[x] for x in out_index_nodes], key=lambda x: x.begin[2])
    print "In nodes: %d, Out nodes: %d" % (len(in_index_nodes), len(out_index_nodes))

    proposed_relations = []
    for out in out_nodes:
        index = index_of_data_with_begin_time_greater_than(in_nodes, out.end[2])
        while index < len(in_nodes):
            current = in_nodes[index]
            if inside_time_cone(out, current, MAX_CELERITY, MAX_TIME, OFFSET):
                score = calculate_score(out, current)
                proposed_relations.append((score, out, current))
            index += 1

    return proposed_relations



def pick_single_best_relation(proposed_relations):
    relations = []
    tmp_proposed_relations = sorted(proposed_relations, key=lambda x: x[0]) # order by score
    while len(tmp_proposed_relations) > 0:
        score, tail, head = tmp_proposed_relations.pop()
        relations.append((score, tail, head))
        tmp_proposed_relations = [x for x in tmp_proposed_relations if x[1] != tail and x[2] != head]

    return relations


def main(ex_id):


result = calculate_relations(BASEDIR, PREFIX)
print "Result: %d" % len(result)

for s, tail, head in result:
    print "%d -> %d" % (tail.id, head.id)
