import os
import sys
import pickle
import itertools
import pickle
import glob
import pandas as pd
import numpy as np


import setpath
os.environ.setdefault('WALDO_SETTINGS', 'default_settings')
from conf import settings
#from wio import Experiment

#import collider
#import wio.file_manager as fm

DATA_DIR = settings.LOGISTICS['filesystem_data']
CHORE_DIR = os.path.abspath('./../data/chore/')
print DATA_DIR


def get_graph(graph_pickle_name):
    print graph_pickle_name
    if not os.path.exists(graph_pickle_name):
        import report_card
        print 'calculating graph'
        graph2, report_df = report_card.collision_iteration2(experiment, graph)
        pickle.dump(graph2, open(graph_pickle_name, 'w'))
    else:
        print 'loading graph.pickle.'
        graph2 = pickle.load(open(graph_pickle_name, 'r'))
    return graph2


def write_blob_file(graph, experiment, basename='chore'):
    file_management = {}
    for i, node in enumerate(graph):

        print node
        node_data = graph.node[node]
        died_f = node_data['died_f']
        if died_f not in file_manager:
            file_manager[died_f] = []

        location = '{f}.{pos}'.format(f=i, pos=0)
        file_manager[died_f].extend([[node, location]])

        df = consolidate_node_data(graph, experiment, node)

    return file_management


def load_summary(ex_dir):
    search_path = os.path.join(ex_dir, '*.summary')
    print 'summary:', glob.glob(search_path)
    # TODO
    summary_path = glob.glob(search_path)[0]
    basename = os.path.basename(summary_path).split('.summary')[0]
    print basename
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.split('%')[0]
        cleaned_lines.append(line.split())
    summary_df = pd.DataFrame(cleaned_lines)
    print summary_df.head()
    return basename, summary_df

def create_lost_and_found(graph):
    lost_and_found = {}
    for i, node in enumerate(graph):
        print node
        node_data = graph.node[node]
        print node_data
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))
        s_data = [[node, 0]]
        p_data = [[0, node]]

        if successors:
            s_data =[[node, s] for s in successors]
        if predecessors:
            p_data = [[p, node] for p in predecessors]

        born_f = node_data['born_f']
        died_f = node_data['died_f']

        if died_f not in lost_and_found:
            lost_and_found[died_f] = []
        if born_f not in lost_and_found:
            lost_and_found[born_f] = []

        lost_and_found[died_f].extend(s_data)
        lost_and_found[born_f].extend(p_data)
    return lost_and_found


if __name__ == '__main__':
    ex_id = '20130318_131111'
    chore_dir = os.path.join(CHORE_DIR, ex_id)
    data_dir  = os.path.join(DATA_DIR, ex_id)
    print chore_dir
    #experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
    #graph = experiment.graph.copy()
    graph_pickle_name = os.path.join(chore_dir, 'graph.pickle')
    graph2 = get_graph(graph_pickle_name)
    #basename, summary_df = load_summary(data_dir)
    basename, summary_df = [], []
    lost_and_found = create_lost_and_found(graph2)

    file_management = write_blob_file(basename=basename, blob_data=blob_df)
