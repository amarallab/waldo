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
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
SHARED_DIR = CODE_DIR + '/shared/'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports

#from wormmetrics.spine_measures import compute_spine_measures
#from wormmetrics.centroid_measures import compute_centroid_measures
#from wormmetrics.basic_measures import compute_basic_measures
#from wormmetrics.basic_measures import compute_width_measures
#from wormmetrics.basic_measures import compute_size_measures
#from database.mongo_retrieve import mongo_query
#from database.mongo_retrieve import unique_blob_ids_for_query
#from database.mongo_insert import compute_and_insert_measurements
#from database.mongo_insert import filter_skipped_and_out_of_range
from wormmetrics.measurement_switchboard import pull_blob_data
from wio.file_manager import write_tmp_file

STANDARD_MEASUREMENTS = ['length', 'centroid_speed_bl', 'curve_bl'] 
# add angle over time

def measure_all(blob_id, store_tmp=True,  measurements=STANDARD_MEASUREMENTS, **kwargs):
    """
    
    Arguments:
    - `blob_id`:
    - `**kwargs`:
    """
    for metric in measurements:        
        t, data = pull_blob_data(blob_id, metric=metric, **kwargs)
        if store_tmp:
            write_tmp_file(blob_id, data_type=metric, data={'time':t, 'data':data})
        
'''
def measure_all_depreciated(blob_id, **kwargs):
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
'''

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
