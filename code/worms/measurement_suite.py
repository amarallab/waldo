#!/usr/bin/env python

'''
Filename: measurement_suite.py
Description: calls all subfunctions required to make all the types of measurements.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys

# path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
shared_directory = project_directory + 'code/shared/'
assert os.path.exists(shared_directory), 'code directory not found'
sys.path.append(shared_directory)

# nonstandard imports
from WormMetrics.spine_measures import compute_spine_measures
from WormMetrics.centroid_measures import compute_centroid_measures
#from WormMetrics.basic_measures import compute_basic_measures
from WormMetrics.basic_measures import compute_width_measures
from WormMetrics.basic_measures import compute_size_measures
from database.mongo_retrieve import mongo_query
from database.mongo_retrieve import unique_blob_ids_for_query
from database.mongo_insert import compute_and_insert_measurements
from database.mongo_insert import filter_skipped_and_out_of_range

# TODO: eliminate redundancy between the following two functions
def insert_stats_for_blob_id(blob_id, metric_computation_function, time_range=[0, 1e20], insert=True, **kwargs):
    measures_dict = metric_computation_function(blob_id, **kwargs)
    for data_type, data_timedict in measures_dict.iteritems():
        times, data = filter_skipped_and_out_of_range(data_timedict, time_range=time_range)
        if insert:
            # insert the newly calculated worm properties into the worm collection
            compute_and_insert_measurements(id=blob_id, data=data, data_name=data_type, time_range=time_range,
                                            col='worm_collection', **kwargs)
    return times, data

def insert_stats(ex_id, metric_computation_function, time_range=[0, 1e20], **kwargs):
    '''
    for each blob_id gather a dictionary of measurments using the metric_computation_function,
    pool all the blob_ids into a combined list.

    '''
    blob_ids = unique_blob_ids_for_query({'ex_id': ex_id})
    plate_metric_dict = {}
    for blob_id in blob_ids:
        metric_dict = metric_computation_function(blob_id)
        for data_type, data_timedict in metric_dict.iteritems():
            times, data = filter_skipped_and_out_of_range(data_timedict, time_range=[])

            # insert the newly calculated worm properties into the worm collection
            compute_and_insert_measurements(id=blob_id, data=data, data_name=data_type, time_range=time_range,
                                            col='worm_collection', **kwargs)

            if data_type not in plate_metric_dict:
                plate_metric_dict[data_type] = []
                # they are both lists, so they should be joined by this action
            plate_metric_dict[data_type] += data

    for data_type, data_list in plate_metric_dict.iteritems():
        compute_and_insert_measurements(id=ex_id, data=data_list, data_name=data_type, time_range=time_range,
                                        col='plate_collection', **kwargs)


def measure_all(blob_id, **kwargs):
    """
    """
    try:
        insert_stats_for_blob_id(blob_id, compute_centroid_measures, **kwargs)
        print 'centroid insert -- Success'
    except Exception as e:
        print 'centroid insert  -- Fail <---------\n{err}'.format(err=e)

    try:
        insert_stats_for_blob_id(blob_id, compute_spine_measures, **kwargs)
        print 'spine insert -- Success'
    except Exception as e:
        print 'spine insert  -- Fail <---------\n{err}'.format(err=e)

    try:
        insert_stats_for_blob_id(blob_id, compute_width_measures, **kwargs)
        insert_stats_for_blob_id(blob_id, compute_size_measures, **kwargs)
        print 'basic measures insert -- Success'
    except Exception as e:
        print 'basic measure insert  -- Fail <---------\n{err}'.format(err=e)

def measure_all_for_ex_id(ex_id, **kwargs):
    blob_ids = unique_blob_ids_for_query({'ex_id': ex_id, 'data_type': 'smoothed_spine'}, **kwargs)
    print len(blob_ids), 'blob ids found'
    for blob_id in blob_ids:
        print blob_id
        #measure_all_hardcore_mode(blob_id)
        measure_all(blob_id, **kwargs)


if __name__ == '__main__':
    if len(sys.argv) < 2:

        # type 1 bug. missing times: 20130318_142605_00201
        blob_ids = ['20130318_142605_00201']
        blob_ids = ['20130324_115435_04452']
        print len(blob_ids)
        for blob_id in blob_ids[:2]:
            measure_all(blob_id)
    else:
        ex_ids = sys.argv[1:]
        for ex_id in ex_ids[:]:
            print 'searching for blobs for', ex_id
            measure_all_for_ex_id(ex_id)
