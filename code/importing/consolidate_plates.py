#!/usr/bin/env python
'''
Filename: consolidate_plates.py
Description:

either: 
grabs relevant info to plot quartiles for each plate across time.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import numpy as np
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(os.path.join(CODE_DIR, '..'))
SHARED_DIR = os.path.join(CODE_DIR, 'shared')

sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from wio.plate_utilities import get_plate_files,  \
     write_dset_summary, \
     return_flattened_plate_timeseries, organize_plate_metadata

from metrics.measurement_switchboard import pull_blob_data, \
     FULL_SET, STANDARD_MEASUREMENTS
from metrics.compute_metrics import quantiles_for_data
from wio.file_manager import get_good_blobs, get_dset
from wio.file_manager import get_timeseries, get_metadata, write_timeseries_file
from wio.file_manager import write_table, read_table
     
def consolidate_plate_timeseries(blob_ids, metric, return_array=True):
    """ this function joins timeseries from all blob_ids and returns the results.
    by default, this is two lists. times and data (a list of lists containin values
    for all blobs at that given time.

    params
    blob_ids: (list)
       the list of all blob ids that should be joined into the plate timeseries
    metric: (str)
       a string denoting what data_type the timeseries are.
    return_array: (bool)
       if false, this returns a dictionary rather than the default output.
    """
    data_dict = {}
    for blob_id in blob_ids:
        # calculate metric for blob, skip if empty list returned        
        btimes, bdata = pull_blob_data(blob_id, metric=metric)
        if len(bdata) == 0:
            continue
        # store blob values in comiled dict.
        for t, value in izip(btimes, bdata):
            new_key = round(t, ndigits=1)
            if new_key not in data_dict:
                data_dict[new_key] = []
            data_dict[new_key].append(value)    


    if not return_array:
        return data_dict        

    times, data = [], []
    N_cols = 0
    for t in sorted(data_dict):
        row = data_dict[t]
        if len(row) > 0:
            times.append(t)
            data.append(row)
            if len(row) > N_cols:
                N_cols = len(row)
        
    # fill out array with NAs        
    filled_data = []
    for d in data:
        d = d + [np.nan] * N_cols
        d = d[:N_cols]
        filled_data.append(d)

    times = np.array(times, dtype=float)
    data = np.array(filled_data, dtype=float)
    return times, data

# '''
# def write_plate_percentiles_old(ex_id, blob_ids=[], metrics=FULL_SET, **kwargs):
#     if not blob_ids:
#         #blob_ids = get_blob_ids(query={'ex_id':ex_id}, **kwargs)
#         blob_ids = get_good_blobs(ex_id)
#     if not blob_ids:
#         return
#
#     #metadata = get_metadata(ID=blob_ids[0], **kwargs)
#     #dataset = metadata.get('dataset', 'none')
#     dataset = get_dset(ex_id)
#
#     plate_dataset = {}
#     bad_blobs = []
#     for bID in blob_ids:
#         blob_data = []
#         blob_is_good = True
#         for metric in metrics:
#             times, data = pull_blob_data(bID, metric=metric)
#             if type(data) == None or len(data) == 0:
#                 print bID, metric, 'not found'
#                 blob_is_good = False
#                 break
#             quantiles = quantiles_for_data(data)
#             if any(np.isnan(quantiles)):
#                 blob_is_good = False
#                 print bID, metric, 'quantiles bad'
#                 break
#             blob_data.extend(quantiles)
#         if blob_is_good:
#             plate_dataset[bID] = blob_data
#         else:
#             bad_blobs.append(bID)
#
#     print len(blob_ids), 'all'
#     print len(bad_blobs), 'bad'
#
#     ids, data = plate_dataset.keys(), plate_dataset.values()
#     # even though this is not writing a timeseries, the format is the same.
#     write_timeseries_file(ID=ex_id,
#                           ID_type='plate',
#                           times=ids,
#                           data=data,
#                           data_type='percentiles',
#                           dset=dataset,
#                           file_tag='worm_percentiles')
# '''
# check to see if ids get written ok
# '''
#     ids2, data2 = get_timeseries(ID=ex_id,
#                                  ID_type='plate',
#                                  times=ids,
#                                  data=data,
#                                  data_type='percentiles',
#                                  dset=dataset,
#                                  file_tag='worm_percentiles')
#
#     print ids2
#     #for i1, i2 in zip(ids, ids2):
#     #    print i1, i2
#     for d1, d2 in zip(data, data2):
#         print d1, d2
# '''


def write_plate_percentiles(ex_id, blob_ids=[], metrics=FULL_SET, **kwargs):
    if not blob_ids:
        #blob_ids = get_blob_ids(query={'ex_id':ex_id}, **kwargs)    
        blob_ids = get_good_blobs(ex_id)
    if not blob_ids:
        return

    
    plate_dataset = {}
    bad_blobs = []
    metric_index = []

    for bID in blob_ids:
        blob_data = []
        blob_is_good = True
        metric_labels = []
        for metric in metrics:
            times, data = pull_blob_data(bID, metric=metric)
            if type(data) == None or len(data) == 0:
                print bID, metric, 'not found'
                blob_is_good = False
                break
            Qs=range(10,91, 10)
            quantiles = quantiles_for_data(data, quantiles=Qs)            
            if any(np.isnan(quantiles)):
                blob_is_good = False
                print bID, metric, 'quantiles bad'
                break
            blob_data.extend(quantiles)
            metric_labels.extend(['{m}-{q}'.format(m=metric, q=q)
                                 for q in Qs])

        if blob_is_good:
            plate_dataset[bID] = blob_data
            metric_index = metric_labels
        else:
            bad_blobs.append(bID)

    print metric_index
    print len(blob_ids), 'all'
    print len(bad_blobs), 'bad'

    percentiles = pd.DataFrame(plate_dataset, index=metric_index)
    percentiles = percentiles.T
    #print percentiles.head()
    #percentiles.to_csv('perc_test.csv')
    write_table(ID=ex_id,
                ID_type='plate',
                dataframe=percentiles,
                data_type='percentiles',
                dset=get_dset(ex_id),
                file_tag='worm_percentiles')

    # '''
    # p2 = read_table(ID=ex_id,
    #                 ID_type='plate',
    #                 data_type='percentiles',
    #                 dset=get_dset(ex_id),
    #                 file_tag='worm_percentiles')
    #
    # print p2.head()
    # '''

def write_plate_timeseries(ex_id, blob_ids=[], measurements=STANDARD_MEASUREMENTS, **kwargs):
    if not blob_ids:
        #blob_ids = get_blob_ids(query={'ex_id':ex_id}, **kwargs)    
        blob_ids = get_good_blobs(ex_id)
    if not blob_ids:
        return
    #metadata = get_metadata(ID=blob_ids[0], **kwargs)    
    #dataset = metadata.get('dataset', 'none')

    dataset = get_dset(ex_id)
    for metric in measurements:
        times, data = consolidate_plate_timeseries(blob_ids, metric, return_array=True)
        write_timeseries_file(ID=ex_id,
                              ID_type='plate',
                              times=times,
                              data=data,
                              data_type=metric,
                              dset=dataset,
                              file_tag='timeseries')                              

if __name__ == '__main__':
    dataset = 'disease_models'
    #data_type = 'cent_speed_bl'
    #data_type = 'length_mm'
    #data_type = 'curve_bl'
    eID = '20131211_145827'
    eID = '20130414_140704'
    #write_plate_timeseries(ex_id=eID)
    metrics = FULL_SET[:]
    write_plate_percentiles(ex_id=eID, metrics=metrics)
