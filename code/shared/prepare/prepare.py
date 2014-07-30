from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

import os
import errno
import pathlib
import pickle

import numpy as np
import pandas as pd

from conf import settings
import multiworm
import wio.file_manager as fm
import collider

__all__ = ['summarize']

DATA_DIR = settings.LOGISTICS['filesystem_data']

def experiment_path(ex_id):
    return os.path.join(DATA_DIR, ex_id)

def graph_cache(ex_id, num_repeats=2):
    print('load experiment')
    ep = experiment_path(ex_id)
    experiment = multiworm.Experiment(experiment_id=ep)
    experiment.load_summary(graph=True)
    graph = experiment.collision_graph

    for i in range(num_repeats):
        #remove nodes outside roi
        print('removing nodes outside roi')
        collider.remove_nodes_outside_roi(graph, experiment, **fm.Preprocess_File(ex_id=ex_id).roi())


        print('round {i}'.format(i=i))
        params = {'offshoots': 20, 'splits_abs': 5, 'splits_rel': 0.5}
        collider.removal_suite(graph, **params)
        # removal suite preforms the following actions:
        # remove_single_descendents(digraph)
        # remove_fission_fusion(digraph, max_split_frames=params_local['splits_abs'])
        # remove_fission_fusion_rel(digraph, split_rel_time=params_local['splits_rel'])
        # remove_offshoots(digraph, threshold=params_local['offshoots'])
        # remove_single_descendents(digraph)

        print('resolving collisions')
        # threshold = 2
        # suspects = collider.suspected_collisions(graph, threshold)
        # print(len(suspects), 'suspects found')
        # collider.resolve_collisions(graph, experiment, suspects)

    # save cache of the graph.
    cache_file = pathlib.Path() / '{}_graphcache2.pkl'.format(ex_id)
    pickle.dump(graph, cache_file.open('wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('Caching graph')
    return graph

def quick_check(ex_id):

    ep = experiment_path(ex_id)
    experiment = multiworm.Experiment(experiment_id=ep)
    experiment.load_summary(graph=True)
    graph = experiment.collision_graph
    all_nodes = set(graph.nodes())

    collider.remove_nodes_outside_roi(graph, experiment, **fm.Preprocess_File(ex_id=ex_id).roi())
    inodes = set(graph.nodes())
    onodes = all_nodes - inodes
    print('all nodes {}'.format(len(all_nodes)))
    print('in nodes {}'.format(len(inodes)))
    print('out nodes {}'.format(len(onodes)))

    roi = check_roi(ex_id)
    rinside = list(roi[roi['inside_roi'] == True]['bid'])
    routside = list(roi[roi['inside_roi'] == False]['bid'])

    print(rinside[:10])
    print('in', len(rinside))
    print('out', len(routside))

    in_mismatches = set(rinside).difference(inodes)
    print('in mismatches', len(in_mismatches))
    print('in mismatches', len(set(inodes).difference(rinside)))


    out_mismatches = set(routside).difference(onodes)
    print('out mismatches', len(out_mismatches))
    print(len(onodes) - len(routside))



def summarize(ex_id):
    # load experiment
    ep = experiment_path(ex_id)
    experiment = multiworm.Experiment(experiment_id=ep)
    experiment.load_summary(graph=True)

    # process the basic blob data
    bounds, terminals, sizes = preprocess_blobs_data(experiment)

    # save it out
    prep_data = fm.PrepData(ex_id)
    prep_data.dump(data_type='bounds', dataframe=bounds, index=False)
    prep_data.dump(data_type='terminals', dataframe=terminals, index=False)
    prep_data.dump(data_type='sizes', dataframe=sizes, index=False)

    roi = check_roi(ex_id, bounds=bounds)
    prep_data.dump(data_type='roi', dataframe=roi, index=False)

    moved = bodylengths_moved(ex_id, bounds=bounds, sizes=sizes)
    prep_data.dump(data_type='moved', dataframe=moved, index=False)

    return bounds, terminals, sizes
