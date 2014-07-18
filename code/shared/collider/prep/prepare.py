from __future__ import print_function, absolute_import, unicode_literals, division
import six
from six.moves import (zip, filter, map, reduce, input, range)

import os
import errno
import pathlib
import pickle

import numpy as np
import pandas as pd

import multiworm
from settings.local import LOGISTICS
import wio.file_manager as fm
import collider

DATA_DIR = LOGISTICS['filesystem_data']

def experiment_path(ex_id):
    return os.path.join(DATA_DIR, ex_id)

def graph_cache(ex_id):
    print('load experiment')
    ep = experiment_path(ex_id)
    experiment = multiworm.Experiment(experiment_id=ep)
    experiment.load_summary(graph=True)
    graph = experiment.collision_graph

    cache_file = pathlib.Path() / '{}_graphcache.pkl'.format(ex_id)
    print('removing nodes outside roi')
    collider.remove_nodes_outside_roi(graph, experiment, **fm.Preprocess_File(ex_id=ex_id).roi())

    print('round 1')
    params = {'offshoots': 20, 'splits_abs': 5, 'splits_rel': 0.5}
    collider.removal_suite(graph, **params)
    print('resolving collisions')
    threshold = 2
    suspects = collider.suspected_collisions(graph, threshold)
    print(len(suspects), 'suspects found')
    collider.resolve_collisions(graph, experiment, suspects)

    print('round 2')
    params = {'offshoots': 20, 'splits_abs': 5, 'splits_rel': 0.5}
    collider.removal_suite(graph, **params)
    print('resolving collisions')
    suspects = collider.suspected_collisions(graph, threshold)
    print(len(suspects), 'suspects found')
    collider.resolve_collisions(graph, experiment, suspects)

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

def _midline_length(points):
    """
    Calculates the length of a path connecting *points*.
    """
    x, y = zip(*points)
    dx = np.diff(np.array(x))
    dy = np.diff(np.array(y))
    return np.sqrt(dx**2 + dy**2).sum()

def check_bounds_against_roi(bounds, x, y, r):
    new = bounds.copy()
    box_x = (new['x_min'] + new['x_max']) / 2
    box_y = (new['y_min'] + new['y_max']) / 2
    new['inside_roi'] = r > np.sqrt((box_x - x)**2 + (box_y - y)**2)
    #print(new.head())
    return new[['bid', 'inside_roi']]

def bodylengths_moved(ex_id, bounds=None, sizes=None):
    bounds_ok = isinstance(bounds, pd.DataFrame)
    sizes_ok = isinstance(sizes, pd.DataFrame)
    if not bounds_ok or not sizes_ok:
        prep_data = fm.PrepData(ex_id)
        if not bounds_ok:
            bounds = prep_data.load('bounds')
        if not sizes_ok:
            sizes = prep_data.load('sizes')


    #print('size\n', sizes.head())
    #print('bound\n', bounds.head())

    moved = pd.concat([bounds.set_index('bid'),
                       sizes.set_index('bid')], axis=1)
    moved.reset_index(inplace=True)
    #print(moved.head())
    dx = bounds['x_max'] - bounds['x_min']
    dy = bounds['y_max'] - bounds['y_min']
    box_diag = np.sqrt(dx**2 + dy**2)
    moved['bl_moved'] = box_diag / moved['midline_median']
    #print(moved.head())
    return moved[['bid', 'bl_moved']]

def preprocess_blobs_data(experiment):
    """


    """
    bounds_data, terminals_data, sizes_data = [], [], []
    for bid, blob in experiment.blobs():
        try:
            if blob['centroid']:
                centroid_x, centroid_y = zip(*blob['centroid'])

                x_min, x_max = min(centroid_x), max(centroid_x)
                y_min, y_max = min(centroid_y), max(centroid_y)
                bounds_data.append({'bid': bid, 'x_min':x_min,
                                    'x_max': x_max, 'y_min':y_min,
                                    'y_max': y_max})

                x0, y0 = blob['centroid'][0]
                xN, yN = blob['centroid'][-1]
                terminals_data.append({'bid': bid, 'x0':x0, 'xN':xN, 'y0':y0, 'yN':yN})

            midline_median, area = np.nan, np.nan
            if blob['midline']:
                midline_median = np.median([_midline_length(p) for p in blob['midline'] if p])
                #sizes.midline_median[bid] = np.median([_midline_length(p) for p in blob['midline'] if p])
            if blob['area']:
                area = np.median(blob['area'])
                #sizes.area_median[bid] = np.median(blob['area'])
            if blob['midline'] or blob['area']:
                sizes_data.append({'bid':bid, 'area_median':area, 'midline_median':midline_median})

        except KeyError:
            # zero frame blobs
            assert blob.blob_data == {}
            pass

    bounds = pd.DataFrame(bounds_data)
    terminals = pd.DataFrame(terminals_data)
    sizes = pd.DataFrame(sizes_data)
    return bounds, terminals, sizes

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
    prep_data.dump(data_type='roi1', dataframe=roi, index=False)

    moved = bodylengths_moved(ex_id, bounds=bounds, sizes=sizes)
    prep_data.dump(data_type='moved1', dataframe=moved, index=False)

    return bounds, terminals, sizes

def check_roi(ex_id, bounds=None):
    # calculate if blob in roi and how many bl each blob moved.
    iprep = fm.Preprocess_File(ex_id=ex_id).roi()
    print(iprep)
    if not isinstance(bounds, pd.DataFrame):
        prep_data = fm.PrepData(ex_id)
        bounds = prep_data.load('bounds')
    print(bounds.head())
    roi = check_bounds_against_roi(bounds, **iprep)
    print(roi.head(20))
    return roi
