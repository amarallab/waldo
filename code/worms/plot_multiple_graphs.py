#!/usr/bin/env python

'''
Filename: plot_multiple_graphs.py
Description: creates a full set of two types of graphs for a list of data_types.

graphs:
1d - scattered, horizontal, bargraph style, dot plot.
2d - typical scatterplot with distributions drawn along the side.
'''

# standard imports
import os
import sys
import datetime

# set paths
from WormProperties.Code.plot_bargraphs import data_for_query

project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
assert os.path.exists(project_directory), 'code directory not found'
sys.path.append(project_directory)
default_graph_directory = os.path.dirname(os.path.realpath(__file__)) + '/../Results/'
bargraph_dir = default_graph_directory + 'Bargraphs/'
scatter_dir = default_graph_directory + 'Scatter/'

# nonstandard imports
from Shared.Code.Plotting.TraitAggregation import scatterplot_2d
from Shared.Code.Database.mongo_retrieve import mongo_query
from Shared.Code.Plotting.TraitAggregation.results_bargraph import table_boxplot

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

data_types = ['smooth_length',
              'curvature_all',
              'speed_perp',
              'speed_along',
              'centroid_speed',
              'centroid_ang_dt',
              'width_mm',
              'size_mm2',
]

data_types_bl = ['smooth_length',
                 'curvature_all_bl',
                 'speed_perp_bl',
                 'speed_along_bl',
                 'centroid_speed_bl',
                 'centroid_ang_dt',
                 'width_bl',
                 'size_mm2']

all_curvatoures = ['curvature_all_bl',
                   'curvature_all',
                   'curvature_head_bl',
                   'curvature_head',
                   'curvature_mid_bl',
                   'curvature_mid',
                   'curvature_tail_bl',
                   'curvature_tail']

size_types = ['smooth_length', 'width_bl', 'size_mm2', 'width_mm']


def plot_bargraphs_for_similar_videos(graph_name, query, data_types=data_types_bl, **kwargs ):
    # make a list of all ex_ids (ie. videos) that
    ex_ids = list(set([e['ex_id'] for e in mongo_query(query, {'ex_id': 1}, col='result_collection', **kwargs)]))
    instructions = [(ex_id, {'ex_id': ex_id}) for ex_id in sorted(ex_ids)]

    for data_type in data_types:
        bargraph_general_processing(data_type, instructions=instructions,
                                            graph_name=graph_name + '_' + data_type)


def make_all_results_bargraphs(instructions, dtypes=data_types_bl):
    for data_type in dtypes:
        print 'results bar-graph for', data_type
        bargraph_general_processing(data_type, instructions=instructions)

    '''
    for i, dt1 in enumerate(dtypes):
        for dt2 in data_types[i + 1:]:
            print dt1, dt2
            scatterplot_2d.general_preprocess(dt1, dt2, instructions=instructions)
    '''

def bargraph_general_processing(data_type, graph_tag='aging', measurement_type='mean', instructions=[],
                                dirname=bargraph_dir, graph_name=''):

    if not graph_name:
        graph_name = data_type + '_' + graph_tag + '_' + str(datetime.date.today())
    savename = dirname + graph_name + '.png'
    print savename
    datasets, labels = [], []
    for (label, query) in instructions:
        datasets.append(data_for_query(query, data_type, measurement_type))
        labels.append(label)
    x_label = str(measurement_type) + ' ' + str(data_type)
    print labels
    print len(datasets)
    print x_label
    print savename
    table_boxplot(labels, datasets, x_label=x_label, savename=savename)


def aging_compare():
    # is this used?
    ages = ['A{n}'.format(n=i) for i in range(1, 14)]
    general_query = {'purpose': 'aging_N2'}
    '''
    instructions = [('A1', {'strain': 'N2', 'age': 'A1', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A2', {'strain': 'N2', 'age': 'A2', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A3', {'strain': 'N2', 'age': 'A3', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A4', {'strain': 'N2', 'age': 'A4', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A5', {'strain': 'N2', 'age': 'A5', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A6', {'strain': 'N2', 'age': 'A6', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A7', {'strain': 'N2', 'age': 'A7', 'ex_id': {'$gt': '20130318_105552'}})
                        ('A8', {'strain': 'N2', 'age': 'A8', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A9', {'strain': 'N2', 'age': 'A9', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A10', {'strain': 'N2', 'age': 'A10', 'ex_id': {'$gt': '20130318_105552'}}),
                        ('A11', {'strain': 'N2', 'age': 'A11', 'ex_id': {'$gt': '20130318_105552'}})
        ]
    '''


def cameara_compare():
    """
    """


    instructions = [('curly', {'strain':'N2', 'age':'A2','ex_id':{'$gt':'20130318_105552'},
                               'source_camera':'curly'}),
                    ('larry', {'strain':'N2', 'age':'A2','ex_id':{'$gt':'20130318_105552'},
                               'source_camera':'larry'}),
                    ('moe', {'strain':'N2', 'age':'A2','ex_id':{'$gt':'20130318_105552'},
                             'source_camera':'moe'})]
   
if __name__ == '__main__':
    #make_all_results_bargraphs(all_curvatoures)
    make_all_results_bargraphs(data_types_bl)
    #plot_bargraphs_for_similar_videos(graph_name='A2', query={'age': 'A2'})


