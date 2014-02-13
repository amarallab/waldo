#!/usr/bin/env python

'''
Filename: plot_scatterplots.py
Description: 

graphs:
1d - scattered, horizontal, bargraph style, dot plot.
2d - typical scatterplot with distributions drawn along the side.
'''
from WormProperties.Code.plot_bargraphs import data_for_query

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import datetime

# Path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
assert os.path.exists(project_directory), 'code directory not found'
sys.path.append(project_directory)

# nonstandard imports
from Shared.Code.Plotting.TraitAggregation import scatterplot_2d
from Shared.Code.Database.mongo_retrieve import mongo_query
#from Shared.Code.Plotting.TraitAggregation.scatterplot_2d import multi_query_plot
from Shared.Code.Plotting import utilities as ut
from Shared.Code.Plotting.TraitAggregation.scatterplot_2d import scatterplot
from Shared.Code.Plotting.TraitAggregation.results_bargraph import table_boxplot

# Globals
GRAPH_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Results/Scatterplots/'

DATA_TYPES = ['smooth_length', 'curvature_all', 'speed_perp',
              'speed_along', 'centroid_speed', 'centroid_ang_ds',
              'width_mm', 'size_mm2']

DATA_TYPES_BL = ['smooth_length', 'curvature_all_bl', 'speed_perp_bl',
                 'speed_along_bl', 'centroid_speed_bl', 'centroid_ang_ds',
                 'width_bl', 'size_mm2']
                                   
def scatterplot_preprocess(data_type1, data_type2, measure1, measure2,
                           data_queries, labels, graph_tag, dirname=GRAPH_DIR):
    """
    :params data_type1, data_type2:
    :params measure1, measure2:
    :param data_queries:
    :param labels:
    :param graph_tag:
    :param dir_name:
    """
    
    save_name = '{dir}{gt}_{dt1}_v_{dt2}_{date}'.format(dir=dirname, 
                                                        gt=graph_tag, dt1=data_type1,
                                                        dt2=data_type2, date=datetime.date.today())
    print save_name    
    #multi_query_plot(queries, labels, data_type1, data_type2, measure1, measure2,
    #                  save_name)
    key1 = '{0}_{1}'.format(data_type1, measure1)
    key2 = '{0}_{1}'.format(data_type2, measure2)
    dsets = [ut.get_multiple_matched_results(q, [key1, key2])
             for q in queries]
    scatterplot(data_sets=dsets, labels=labels, x_label=key1, y_label=key2, savename=save_name)

    
def compare_data_types_pairwise(data_queries, labels, dtypes=DATA_TYPES_BL):
    '''
    generates plots for each pairwise comparision of two data types
    by plotting the mean values for both types against each other.
    The data from multiple different queries can be plotted on the same graph.

    parameters:
    :param labels: a list of strings to label each of the queries on the graphs
    :param data_queries: a list of dicts in database query format to retrieve one dataset.
    :param dtypes: a list of data_types to be compaired pairwise.
    '''    
    for i, dt1 in enumerate(dtypes):
        for dt2 in dtypes[i + 1:]:
            print dt1, dt2
            scatterplot_preprocess(data_type1=dt1, data_type2=dt2,
                                   measure1='mean', measure2='mean',
                                   data_queries=data_queries, labels=labels,
                                   graph_tag='compare')

def compare_mean_vs_std(data_queries, labels, dtypes=DATA_TYPES_BL):
    '''
    generates plot comparing the mean vs the standard deviation for each data type in dtypes.
    for both types against each other. Multiple datasets can be pulled from the worm database
    and plotted against one another.

    parameters:
    :param labels: a list of strings to label each of the queries on the graphs
    :param data_queries: a list of dicts in database query format to retrieve one dataset.
    :param dtypes: a list of data_types to be plotted
    '''    
    for dt in dtypes:
        scatterplot_preprocess(data_type1=dt, data_type2=dt,
                               measure1='mean', measure2='std',
                               data_queries=data_queries, labels=labels,
                               graph_tag='std_v_mean')
    
            
def compare_duration_vs_data_type(data_queries, labels, dtypes=DATA_TYPES_BL):
    '''
    generates plot comparing the mean vs the standard deviation for each data type in dtypes.
    for both types against each other. Multiple datasets can be pulled from the worm database
    and plotted against one another.

    parameters:
    :param labels: a list of strings to label each of the queries on the graphs
    :param data_queries: a list of dicts in database query format to retrieve one dataset.
    :param dtypes: a list of data_types to be plotted
    '''    

    for dt in dtypes:
        key = '{0}_mean'.format(dt)
        save_name = '{dir}duration_v_{dt}_{date}'.format(dir=GRAPH_DIR, dt=dt,
                                                         date=datetime.date.today())
        dsets = [ut.pair_metavalue_vs_datavalue(query, data_key=key, 
                                                meta_key='duration')
                 for query in queries]
        print save_name    
        scatterplot(data_sets=dsets, labels=labels, x_label='duration', y_label=key,
                    savename=save_name)
            
def test_data():
    """
    """
    age_labels = ['A{0}'.format(n) for n in xrange(1, 5)]
    base_query = {'strain': 'N2'}
    queries = []
    for age in age_labels:
        q = {'age': age}
        q.update(base_query)
        queries.append(q)
    return age_labels, queries
                    
if __name__ == '__main__':
    labels, queries = test_data()
    #compare_data_types_pairwise(data_queries=queries, labels=labels)
    #compare_mean_vs_std(data_queries=queries, labels=labels)
    compare_duration_vs_data_type(data_queries=queries, labels=labels)
