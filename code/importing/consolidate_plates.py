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
import matplotlib.pyplot as plt
import scipy.stats as stats

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

from wormmetrics.measurement_switchboard import pull_blob_data, quantiles_for_data, \
     FULL_SET, STANDARD_MEASUREMENTS
from wio.file_manager import get_blob_ids, get_timeseries, get_metadata, write_timeseries_file
     
def consolidate_plate_timeseries(blob_ids, metric, return_array=True):
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

    if return_array:
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
    else:
        return data_dict        

def write_plate_percentiles(ex_id, blob_ids=[], metrics=FULL_SET, **kwargs):
    if not blob_ids:
        blob_ids = get_blob_ids(query={'ex_id':ex_id}, **kwargs)    
    metadata = get_metadata(ID=blob_ids[0], **kwargs)    
    dataset = metadata.get('dataset', 'none')
    plate_dataset = {}
    bad_blobs = []
    for bID in blob_ids:
        blob_data = []
        blob_is_good = True
        for metric in metrics:
            times, data = pull_blob_data(bID, metric=metric)
            if type(data) == None or len(data) == 0:
                print bID, metric, 'not found'
                blob_is_good = False
                break
            quantiles = quantiles_for_data(data)
            if any(np.isnan(quantiles)):
                blob_is_good = False
                print bID, metric, 'quantiles bad'
                break
            blob_data += quantiles
        if blob_is_good:
            plate_dataset[bID] = blob_data
        else:
            bad_blobs.append(bID)

    print len(blob_ids), 'all'
    print len(bad_blobs), 'bad'
            
    ids, data = plate_dataset.keys(), plate_dataset.values()
    write_timeseries_file(ID=ex_id,
                          ID_type='plate',
                          times=ids,
                          data=data,
                          data_type='percentiles',
                          dset=dataset,
                          file_tag='worm_percentiles')
    # check to see if ids get written ok
    '''
    ids2, data2 = get_timeseries(ID=ex_id,
                                 ID_type='plate',
                                 times=ids,
                                 data=data,
                                 data_type='percentiles',
                                 dset=dataset,
                                 file_tag='worm_percentiles')                              

    print ids2
    #for i1, i2 in zip(ids, ids2):
    #    print i1, i2
    for d1, d2 in zip(data, data2):
        print d1, d2
    '''

def write_plate_timeseries_set(ex_id, blob_ids=[], measurements=STANDARD_MEASUREMENTS, **kwargs):
    if not blob_ids:
        blob_ids = get_blob_ids(query={'ex_id':ex_id}, **kwargs)    
    metadata = get_metadata(ID=blob_ids[0], **kwargs)    
    dataset = metadata.get('dataset', 'none')
    for metric in measurements:
        '''
        path_tag = '{ds}-{m}'.format(ds=dataset, m=metric)
        print path_tag        
        write_full_plate_timeseries(ex_id=ex_id,
                                    metric=metric,
                                    path_tag=path_tag,
                                    blob_ids=blob_ids,
                                    **kwargs)
        '''
        times, data = consolidate_plate_timeseries(blob_ids, metric, return_array=True)
        write_timeseries_file(ID=ex_id,
                              ID_type='plate',
                              times=times,
                              data=data,
                              data_type=metric,
                              dset=dataset,
                              file_tag='timeseries')                              
    
     
def process_basic_plate_timeseries(dataset, data_type, verbose=True):
    
    ex_ids, dfiles = get_plate_files(dataset, data_type)
    means, stds = [], []
    quartiles = []
    hours = []
    days = []
    #labels, sublabels, plate_ids = {}, {}, {}
    labels, sublabels, plate_ids = [], [], []

    for i, (ex_id, dfile) in enumerate(izip(ex_ids, dfiles)):
        hour, label, sub_label, pID, day = organize_plate_metadata(ex_id)
        hours.append(hour)
        labels.append(label)
        sublabels.append(sub_label)
        plate_ids.append(pID)
        days.append(day)

        flat_data = return_flattened_plate_timeseries(ex_id, dataset, data_type)
        if not len(flat_data):
            continue
        means.append(np.mean(flat_data))
        stds.append(np.std(flat_data))
        men = float(np.mean(flat_data))
        #print men, type(men), men<0
        #print flat_data[:5]
        quartiles.append([stats.scoreatpercentile(flat_data, 25),
                          stats.scoreatpercentile(flat_data, 50),
                          stats.scoreatpercentile(flat_data, 75)])
        if verbose:
            print '{i} {eID} | N: {N} | hour: {h} | label: {l}'.format(i=i, eID=ex_id,
                                                                       N=len(flat_data),
                                                                       h=round(hour, ndigits=1), l=label)                                      

    #for i in zip(ex_ids, means, stds, quartiles):
    #    print i
    data={'ex_ids':ex_ids,
          'hours':hours,
          'mean':means,
          'std':stds,
          'quartiles':quartiles,
          'labels':labels,
          'sub':sublabels,
          'plate_ids':plate_ids,
          'days':days,
          }
    return data

def process_basics_for_standard_set(dataset):
    standard_set = ['cent_speed_bl', 'length_mm', 'curve_bl']
    for data_type in standard_set:
        print dataset, data_type,
        data = process_basic_plate_timeseries(dataset, data_type)
        write_dset_summary(data=data, sum_type='basic', 
                           data_type=data_type, dataset=dataset)

if __name__ == '__main__':
    dataset = 'disease_models'
    #data_type = 'cent_speed_bl'
    #data_type = 'length_mm'
    #data_type = 'curve_bl'
    #process_basics_for_standard_set(dataset)
    process_fitting_for_speed(dataset)
