#!/usr/bin/env python

'''
Filename: switchboard.py
Description: 
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import json

# path definitions
project_directory =  os.path.dirname(os.path.realpath(__file__)) + '/../../../'
sys.path.append(project_directory)

# nonstandard imports
from spine_measures import compute_spine_measures
from centroid_measures import compute_centroid_measures
from basic_measures import compute_width_measures
from basic_measures import compute_size_measures
from database.mongo_retrieve import unique_blob_ids_for_query
from database.mongo_retrieve import pull_data_type_for_blob


# globals
# TODO: Figure out a way to build the switchboard from the functions without hard coding in acceptable_types
'''
SWITCHES = {'basic': (compute_basic_measures, ['width_mm', 'width_bl', 'size_mm2']),
            'spine': (compute_spine_measures, ['smooth_length', 
                                               'speed_along', 'speed_along_head', 'speed_along_mid', 'speed_along_tail',
                                               'speed_along_bl', 'speed_along_head_bl', 'speed_along_mid_bl', 'speed_along_tail_bl',
                                               'speed_perp', 'speed_perp_head', 'speed_perp_mid', 'speed_perp_tail',
                                               'speed_perp_bl', 'speed_perp_head_bl', 'speed_perp_mid_bl', 'speed_perp_tail_bl',
                                               'curvature_all', 'curvature_head', 'curvature_mid',
                                               'curvature_tail',
                                               'curvature_all_bl',
                                               'curvature_head_bl',
                                               'curvature_mid_bl',
                                               'curvature_tail_bl']),
            'centroid': (compute_centroid_measures, ['centroid_ang_ds', 'centroid_speed_bl', 'centroid_speed']),
            'eigenworm': (None, []),
            'gaussian': (None, []),
            }
'''
SWITCHES = {'width': {'func': compute_width_measures, 'data_type': 'width50', 'metrics': ['width_mm', 'width_bl']},
             'size': {'func': compute_size_measures, 'data_type': 'size_raw', 'metrics': ['size_mm2']},
             'spine': {'func': compute_spine_measures, 'data_type': 'smoothed_spine',
                       'metrics': ['smooth_length',
                                   'speed_along', 'speed_along_head', 'speed_along_mid', 'speed_along_tail',
                                   'speed_along_bl', 'speed_along_head_bl', 'speed_along_mid_bl', 'speed_along_tail_bl',
                                   'speed_perp', 'speed_perp_head', 'speed_perp_mid', 'speed_perp_tail',
                                   'speed_perp_bl', 'speed_perp_head_bl', 'speed_perp_mid_bl', 'speed_perp_tail_bl',
                                   'curvature_all', 'curvature_head', 'curvature_mid', 'curvature_tail',
                                   'curvature_all_bl', 'curvature_head_bl', 'curvature_mid_bl', 'curvature_tail_bl']},
            'centroid': {'func': compute_centroid_measures, 'data_type': 'xy_raw',
                         'metrics': ['centroid_ang_ds', 'centroid_speed_bl', 'centroid_speed']}}

def switchboard(metric):
    for measure_type, props in SWITCHES.iteritems():
        m_function, acceptable_metrics = props['func'], props['metrics']
        if metric in acceptable_metrics:
            return m_function
    assert False, 'the metric you specified could not be located'

def pull_metric_for_blob_id(blob_id, metric, remove_skips=True, return_timedict=True, **kwargs):
        metric_computation_function = switchboard(metric)
        results = metric_computation_function(blob_id=blob_id, metric=metric, **kwargs)
        if remove_skips:
            new_results = {}
            for k,v in results.iteritems():
                if isinstance(v, float):
                    new_results[k] = v
            # this would be perferable but only works in python 2.7
            #return {k: v for (k, v) in results.iteritems() if isinstance(v, float)}
            return new_results
        return results

def pull_metric_for_ex_id(ex_id, metric, verbose=False):
    # TODO make dependent on pull metric for blob_id
    # find which function you need for that metric and which blob_ids to include on the search
    metric_computation_function = switchboard(metric)
    blob_ids = unique_blob_ids_for_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'})
    # go through every blob and pool the data of that type
    pooled_data = []
    for blob_id in blob_ids[:]:
        new_values = metric_computation_function(blob_id=blob_id, metric=metric).values()
        if verbose:
            print  '%s\tlen:%i\texamples:%s' % (blob_id, len(new_values), str(new_values[:4]))
        pooled_data += new_values
    return pooled_data

def list_all_metrics():
    all_metrics = []
    for measure_type, properties in SWITCHES.iteritems():
        all_metrics += properties['metrics']
    return all_metrics

def pull_all_for_ex_id(ex_id):    
    for metric in list_all_metrics():
        yield metric, pull_metric_for_ex_id(ex_id, metric)

# depreciated version of this code.
'''
def pull_all_for_blob_id(blob_id, out_format='values', **kwargs):
    all_data = {}
    assert out_format in ['values', 'timedict']
    for metric in list_all_metrics():
        if out_format == 'values':
            all_data[metric] = pull_metric_for_blob_id(blob_id, metric, **kwargs).values()
        else:
            all_data[metric] = pull_metric_for_blob_id(blob_id, metric, **kwargs)
    return all_data
'''

def pull_all_for_blob_id(blob_id, out_format='values', **kwargs):
    all_data = {}
    assert out_format in ['values', 'timedict']
    flag_data_entry = pull_data_type_for_blob(blob_id, data_type='flags', **kwargs)
    for metric_type, properties in SWITCHES.iteritems():
        source_data_entry = pull_data_type_for_blob(blob_id, data_type=properties['data_type'], **kwargs)
        for metric in properties['metrics']:
            #print metric
            if out_format == 'values':
                all_data[metric] = pull_metric_for_blob_id(blob_id, metric=metric, source_data_entry=source_data_entry,
                                                           flag_data_entry=flag_data_entry, **kwargs).values()
            else:
                all_data[metric] = pull_metric_for_blob_id(blob_id, metric=metric, source_data_entry=source_data_entry,
                                                           flag_data_entry=flag_data_entry, **kwargs)

    return all_data

if __name__ == '__main__':
    '''
    metric = 'speed_perp'
    metric = 'size_mm2'
    #metric = 'centroid_ang_ds'
    ex_id = '00000000_000001'
    #pull_metric_for_ex_id(ex_id, metric)
    for (m, data) in pull_all_for_ex_id(ex_id):
        print m, len(data)
    '''
    bID = '20130415_104153_00853'
    #bID='00000000_000001_00001'
    import time
    start = time.time()
    pull_all_for_blob_id(blob_id=bID)
    dur1 = time.time() - start
    start = time.time()
    pull_all_for_blob_id2(blob_id=bID)
    dur2 = time.time() - start

    print '1', dur1
    print '2', dur2
    print dur1/dur2
