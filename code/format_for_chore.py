import os
import sys
from glob import glob
import pickle
import itertools

import pandas as pd
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import prettyplotlib as ppl
from statsmodels.distributions.empirical_distribution import ECDF

import setpath
import pickle
import glob
os.environ.setdefault('WALDO_SETTINGS', 'default_settings')

from conf import settings
from wio import Experiment

import collider
import wio.file_manager as fm
import report_card
DATA_DIR = settings.LOGISTICS['filesystem_data']
CHORE_DIR = os.path.abspath('./../data/chore/')
print DATA_DIR


def get_graph(graph_pickle_name):

    print graph_pickle_name
    if not os.path.exists(graph_pickle_name):
        print 'calculating graph'
        graph2, report_df = report_card.collision_iteration2(experiment, graph)
        pickle.dump(graph2, open(graph_pickle_name, 'w'))
    else:
        print 'loading graph.pickle.'
        graph2 = pickle.load(open(graph_pickle_name, 'r'))
    return graph2


def write_blob_file(bid, n, basename, blob_data):
    # TODO write this thing
    pass


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

if __name__ == '__main__':
    ex_id = '20130318_131111'
    chore_dir = os.path.join(CHORE_DIR, ex_id)
    data_dir  = os.path.join(DATA_DIR, ex_id)
    print chore_dir

    experiment = Experiment(experiment_id=ex_id, data_root=DATA_DIR)
    graph = experiment.graph.copy()
    graph_pickle_name = os.path.join(chore_dir, 'graph.pickle')
    #graph2 = get_graph(graph_pickle_name)

    basename, summary_df = load_summary(data_dir)

    lost_and_found = {}
    graph2 = []
    for i, node in enumerate(graph2):
        print node
        node_data = graph.node[node]
        print node_data
        successors = graph.successors(node)
        predecessors = graph.predecessors(node)

        print successors
        print predecessors

        blob_df = []
        write_blob_file(bid=node, n=i, basename=basename, blob_data=blob_df)

        if i > 5:
            break
