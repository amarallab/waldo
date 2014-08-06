__author__ = 'heltena'

#import pathcustomize
import os
import sys

HERE = os.path.abspath('.')
CODE_DIR = os.path.join(HERE, '..')
print CODE_DIR
sys.path.append(CODE_DIR)
import setpath
import platform
import math

print('Python {} ({}) [{}] on {}'.format(platform.python_version(), ', '.join(platform.python_build()),
                                         platform.python_compiler(), sys.platform))

from wio.experiment import Experiment


def dist_2d(c1, c2):
    xc = c1[0] - c2[0]
    yc = c1[1] - c2[1]
    return math.hypot(xc, yc)


def dist_3d(c1, c2):
    d = [c1[i] - c2[i] for i in range(3)]
    return math.sqrt(sum([x * x for x in d]))


# dl is an array of DataRow ordered by "begin time"
def index_of_data_with_begin_time_greater_than(dl, t):
    first = 0
    last = len(dl)
    while abs(first - last) > 1:
        middle = (first + last) / 2
        middle_value = dl.iloc[middle]['t0']
        if middle_value < t:
            first = middle
        else:
            last = middle
    while first < len(dl) and dl.iloc[first]['t0'] <= t:
        first += 1
    return first


def inside_time_cone(tail_pos, head_pos, max_celerity, max_time, offset):
    if tail_pos == head_pos:
        return True
    dt = head_pos[2] - tail_pos[2]
    if dt < 0 or dt >= max_time:
        return False
    radius = offset + max_celerity * dt
    return dist_2d(tail_pos, head_pos) < radius


def calculate_score(tail_pos, head_pos):
    # TODO: Call to Nick score function
    dist = dist_3d(tail_pos, head_pos)
    if dist == 0:
        return 10000.0
    else:
        return 1.0 / dist


def collision_finder(graph, experiment, max_celerity, max_time, offset):
    terminals_df = experiment.prepdata.load('terminals')
    data_index_nodes = set(terminals_df['bid'])
    in_index_nodes = set([x for x in graph.nodes() if len(graph.predecessors(x)) == 0]) & data_index_nodes
    out_index_nodes = set([x for x in graph.nodes() if len(graph.successors(x)) == 0]) & data_index_nodes
    terminals_df['useful_in'] = terminals_df['bid'].apply(lambda x: x in in_index_nodes)
    terminals_df['useful_out'] = terminals_df['bid'].apply(lambda x: x in out_index_nodes)
    in_nodes = terminals_df[terminals_df['useful_in']]
    out_nodes = terminals_df[terminals_df['useful_out']]

    proposed_relations = []
    for count, out in out_nodes.iterrows():
        out_pos = (out['xN'], out['yN'], out['tN'])
        index = index_of_data_with_begin_time_greater_than(in_nodes, out['tN'])
        while index < len(in_nodes):
            current = in_nodes.iloc[index]
            if current['t0'] > out['tN'] + max_time:
                break
            current_pos = (current['x0'], current['y0'], current['t0'])
            if inside_time_cone(out_pos, current_pos, max_celerity, max_time, offset):
                score = calculate_score(out, current)
                proposed_relations.append((score, out, current))
            index += 1

    relations = []
    tmp_proposed_relations = sorted(proposed_relations, key=lambda x: x[0]) # order by score
    while len(tmp_proposed_relations) > 0:
        score, tail, head = tmp_proposed_relations.pop()
        relations.append((score, tail, head))
        tmp_proposed_relations = [x for x in tmp_proposed_relations if x[1].bid != tail.bid and x[2].bid != head.bid]

    # TODO: connect the nodes in the graph
    print len(relations)


if __name__ == "__main__":
    ex_id = '20130318_131111'
    experiment = Experiment(experiment_id=ex_id)
    graph = experiment.collision_graph
    max_celerity = 5  # pixels per second
    max_time = 5  # max 5 seconds in a gap
    offset = 10  # pixels added to the radius of the cone
    collision_finder(graph, experiment, max_celerity, max_time, offset)