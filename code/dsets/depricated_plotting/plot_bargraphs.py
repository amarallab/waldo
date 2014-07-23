#!/usr/bin/env python

'''
Filename: plot_bargraphs.py
Description:

graphs:
1d - scattered, horizontal, bargraph style, dot plot.
2d - typical scatterplot with distributions drawn along the side.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import datetime

# Path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# nonstandard imports
from database.mongo_retrieve import mongo_query
from Plotting.TraitAggregation.results_bargraph import table_boxplot
from Plotting.TraitAggregation.multi_histogram import table_histogram_set
from Plotting import utilities as ut

# Globals
BARGRAPH_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Results/Bargraphs/'

DATA_TYPES = ['smooth_length', 'curvature_all', 'speed_perp',
              'speed_along', 'centroid_speed', 'centroid_ang_ds',
              'width_mm', 'size_mm2']

DATA_TYPES_BL = ['smooth_length', 'curvature_all_bl', 'speed_perp_bl',
                 'speed_along_bl', 'centroid_speed_bl', 'centroid_ang_ds',
                 'width_bl', 'size_mm2']

def bargraph_processing(data_type, graph_tag='aging', measurement_type='mean',
                        labels=[], queries=[], dirname=BARGRAPH_DIR, graph_name=''):
    '''
    Argurments:
    data_type--
    graph_tag--
    measurement_type --
    labels--
    queries--
    dirname--
    graphname --

    '''
    assert len(labels) == len(queries)
    if not graph_name:
        graph_name = '{gt}_{dt}_{date}'.format(gt=graph_tag, dt=data_type,
                                               date=datetime.date.today())
    savename = dirname + graph_name + '.png'
    print savename
    datasets = [data_for_query(query, data_type, measurement_type)
                for query in queries]
    x_label = str(measurement_type) + ' ' + str(data_type)
    table_boxplot(labels, datasets, x_label=x_label, savename=savename)
    savename2 = dirname + 'hist_' + graph_name + '.png'
    try:
        table_histogram_set(datasets=datasets, labels=labels, x_label=x_label, savename=savename2)
    except:
        print savename, 'not written. datasets invalid'

def bargraphs_by_ex_id(query, data_types=DATA_TYPES_BL, **kwargs):
    '''
    Makes a bargraph plot that shows every ex_id seperatly
    '''
    graph_tag = 'ex_ids_for_{q}'.format(q=query)
    ex_id_labels = list({e['ex_id'] for e in mongo_query(query, col='worm_collection', **kwargs)})
    ex_id_labels.sort()
    queries = [{'ex_id': ex_id} for ex_id in ex_id_labels]
    for data_type in data_types:
        bargraph_processing(data_type, labels=ex_id_labels, queries=queries,
                            graph_tag=graph_tag)

    else:
        print 'no worms found for', query


def bargraphs_compare_camera(age='A2', data_types=DATA_TYPES_BL, base_query={}):
    '''
    Makes a bargraph plot that shows every ex_id seperatly
    '''
    graph_tag = 'camera_{age}'.format(age=age)
    base_query.update({'strain': 'N2', 'age': age, 'ex_id':{'$gt':'20130318_105552'}})
    cam_labels = ['curly', 'larry', 'moe']
    queries = []
    for cam in cam_labels:
        q = {'source_camera': cam}
        q.update(base_query)
        queries.append(q)
    for data_type in data_types[:]:
        bargraph_processing(data_type, labels=cam_labels,
                            queries=queries, graph_tag=graph_tag)

def bargraphs_compare_age(age_range=(2,7), data_types=DATA_TYPES_BL):
    '''
    Makes a bargraph plot that shows
    '''
    graph_tag = 'aging'
    #base_query = {'strain': 'N2', 'ex_id': {'$lt': '20130318_153741'}}
    base_query = {'strain': 'N2', 'ex_id': {'$gt': '20130318_153741'}}
    age_labels = ['A{0}'.format(n) for n in xrange(age_range[0], age_range[1]+1)]
    queries = []
    for age in age_labels:
        q = {'age': age}
        q.update(base_query)
        queries.append(q)
    for data_type in data_types[:]:
        bargraph_processing(data_type, labels=age_labels, queries=queries,
                            graph_tag=graph_tag)


def data_for_query(query, data_type, measurement_type):
    assert measurement_type in default_mf
    result_dicts, metadata_dicts = ut.pull_data_from_results_db(query)
    filtered_result_dicts = ut.filter_results_multi(data_type, result_dicts)
    blob_ids, data = ut.results_dict_to_lists(filtered_result_dicts, measurement_type)
    return data

if __name__ == '__main__':
    #bargraphs_compare_camera(age='A2', data_types=['curvature_all'])
    bargraphs_compare_age(age_range=(1, 7))
    '''
    bargraphs_by_ex_id(query={'age':'A2'})
    bargraphs_by_ex_id(query={'age':'A3'})
    bargraphs_by_ex_id(query={'age':'A4'})
    bargraphs_by_ex_id(query={'age':'A5'})
    bargraphs_by_ex_id(query={'age':'A6'})
    bargraphs_by_ex_id(query={'age':'A7'})
    bargraphs_by_ex_id(query={'age':'A8'})
    '''
