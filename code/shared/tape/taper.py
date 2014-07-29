#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import (
        absolute_import, division, print_function, unicode_literals)
import six
from six.moves import (zip, filter, map, reduce, input, range)

# standard library
import math

# third party
import numpy as np

# project specific
from conf import settings
from .scorer import Scorer

# parameters
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
    Initalized with a wio.Experiment-like object and a simplified graph.
    """
    def __init__(self, experiment, graph):#, regenerate_cache=False):
        self.experiment = experiment

        self._scorer = Scorer(experiment)

        self.max_speed = self._scorer.max_speed * settings.TAPE_MAX_SPEED_MULTIPLIER

        self._terminals = self.experiment.prepdata.load('terminals').set_index('bid')
        self.graph = graph

    def score(lost_id, found_id):
        lost = self._terminals.loc[lost_id]
        found = self._terminals.loc[found_id]

        Dx = lost.xN - found.x0
        Dy = lost.yN - found.y0
        Dt = lost.tN - found.t0

        Dr = math.sqrt(Dx**2 + Dy**2)

        return self._scorer(Dr, Dt)

    def find_start_and_end_nodes(self):

        graph = self.graph
        terminals = self._terminals

        # go through graph and find all node ids for start/stop canidates
        start_nodes = [x for x in graph.nodes() if len(graph.successors(x)) == 0]
        stop_nodes = [x for x in graph.nodes() if len(graph.predecessors(x)) == 0]

        # reformat terminals dataframe.
        terms = terminals.copy()
        terms = terms[np.isfinite(terms['t0'])] # drop NaN rows
        term_ids = set(terms['bid']) # get set of all bids with data
        terms.set_index('bid', inplace=True)
        terms['node_id'] = 0

        # make sets that contain bids of all components
        stop_components = []
        for bid in stop_nodes:
            comps = graph[bid].get('components', [bid])
            stop_components.extend(comps)
        stop_components = set(stop_components)

        start_components = []
        for bid in start_nodes:
            comps = graph[bid].get('components', [bid])
            start_components.extend(comps)
        start_components = set(start_components)

        # print(len(term_ids), 'term ids')
        # print(len(start_components), 'start set')
        # print(len(stop_components), 'stop set')
        # print(len(stop_components & start_components), 'start/stop overlap')

        start_bids = start_components & term_ids
        stop_bids = stop_components & term_ids

        # split up terminals dataframe into two smaller ones containing
        # only relevant information.
        terms.reset_index(inplace=True)
        start_terms = terms.loc[list(start_bids)][['bid', 't0', 'x0', 'y0', 'node_id']]
        end_terms = terms.loc[list(stop_bids)][['bid', 'tN', 'xN', 'yN', 'node_id']]

        start_terms.rename(columns={'t0':'t', 'x0':'x', 'y0':'y'}, inplace=True)
        end_terms.rename(columns={'tN':'t', 'xN':'x', 'yN':'y'}, inplace=True)

        # add in the node_id values into the dataframes
        for node_id in start_nodes:
            comps = graph[node_id].get('components', [node_id])
            for comp in comps:
                if comp in start_bids:
                    start_terms['node_id'].loc[comp] = node_id

        for node_id in stop_nodes:
            comps = graph[node_id].get('components', [node_id])
            for comp in comps:
                if comp in stop_bids:
                    end_terms['node_id'].loc[comp] = node_id

        # drop rows with NaN as 't' (most other data will be missing)
        start_terms = start_terms[np.isfinite(start_terms['t'])]
        end_terms = end_terms[np.isfinite(end_terms['t'])]

        # print('dropped NaN values')
        # print(len(start_terms), 'start terms')
        # print(len(end_terms), 'end terms')

        # sort nodes using time.
        start_terms.sort(columns='t', inplace=True)
        end_terms.sort(columns='t', inplace=True)

        # drop rows that have duplicate node_ids.
        # for starts, take the first row (lowest time)
        start_terms.drop_duplicates(subset='node_id', take_last=False,
                                    inplace=True)
        # for ends, take the last row (highest time)
        end_terms.drop_duplicates(subset='node_id', take_last=True,
                                  inplace=True)

        print('removed all but one component for node_ids')
        print(len(start_terms), 'start terms')
        print(len(end_terms), 'end terms')

        return start_terms, end_terms


    def match_start_and_end(self, start_terms, end_terms):
        pass



    #     for comp in comps:
    #         if comp in term_ids:
    #             terms['head_id'].loc[comp] = bid
    #             terms['start'].loc[comp] = True


    # starts = terms[terms['start'] == True]
    # ends = terms[terms['start'] == True][['bid', 'tN', 'xN', 'yN', 'head_id']]


    #data_index_nodes = set(terms['bid'])

    #return starts, ends

    # in_components = ''
    # in_components = ''


    # in_index_nodes = set([x for x in network.nodes() if len(network.predecessors(x)) == 0])# & data_index_nodes
    # out_index_nodes = set([x for x in network.nodes() if len(network.successors(x)) == 0])# & data_index_nodes


    # # TODO: resume recoding here.
    # in_nodes = sorted([data_map[x] for x in in_index_nodes], key=lambda x: x.begin[2])
    # out_nodes = sorted([data_map[x] for x in out_index_nodes], key=lambda x: x.begin[2])
    # print("In nodes: %d, Out nodes: %d" % (len(in_index_nodes), len(out_index_nodes)))

    # proposed_relations = []
    # for out in out_nodes:
    #     index = index_of_data_with_begin_time_greater_than(in_nodes, out.end[2])
    #     while index < len(in_nodes):
    #         current = in_nodes[index]
    #         if inside_time_cone(out, current, MAX_CELERITY, TAPE_FRAME_SEARCH_LIMIT, TAPE_SHAKYCAM_ALLOWANCE):
    #             score = calculate_score(out, current)
    #             proposed_relations.append((score, out, current))
    #         index += 1

    # return proposed_relations

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
    print("Result: %d" % len(result))

    for s, tail, head in result:
        print("%d -> %d" % (tail.id, head.id))
