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
import numpy as np
from itertools import izip

# path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
SHARED_DIR = os.path.abspath(HERE + '/../')
sys.path.append(SHARED_DIR)

# nonstandard imports
from measurement_switchboard import pull_blob_data, quantiles_for_data
from wio.file_manager import write_timeseries_file, get_metadata, get_blob_ids
from wio.file_manager import get_timeseries

#from wio.export_data import write_full_plate_timeseries
# add angle over distance

STANDARD_MEASUREMENTS = ['length_mm', 'curve_w', 'cent_speed_bl']
FULL_SET = ['length_mm', 'width_mm', 'curve_w', 'cent_speed_bl']

def measure_all(blob_id, store_tmp=True,  measurements=FULL_SET, **kwargs):
    """
    Arguments:
    - `blob_id`:
    - `**kwargs`:
    """    
    print 'Computing Standard Measurements'
    # prepare scaling factors to convert from pixels to units
    metadata = get_metadata(blob_id, **kwargs)
    lt, lengths = pull_blob_data(blob_id, metric='length', **kwargs)
    wt, widths = pull_blob_data(blob_id, metric='width50', **kwargs)
    pixels_per_bl = float(np.median(lengths))
    pixels_per_w = float(np.median(widths))
    pixels_per_mm = float(metadata.get('pixels-per-mm', 1.0))
    print '\tunits mm : {mm} | bl: {bl} | w:{w}'.format(bl=round(pixels_per_bl, ndigits=2),
                                                        mm=round(pixels_per_mm, ndigits=2),
                                                        w=round(pixels_per_w, ndigits=2))
    # loop through standard set of measurements and write them
    for metric in measurements:        
        t, data = pull_blob_data(blob_id, metric=metric,
                                 pixels_per_bl=pixels_per_bl,
                                 pixels_per_mm=pixels_per_mm, 
                                 pixels_per_w=pixels_per_w, 
                                 **kwargs)
        if store_tmp:
            write_timeseries_file(blob_id, data_type=metric,
                                  times=t, data=list(data))

        try:
            mean = round(np.mean(data), ndigits=3)
            s = round(np.std(data), ndigits=3)
        except:
            mean, std = 'Na', 'Na'
        N = len(data)
        print '\t{m} mean: {mn} | std: {s} | N: {N}'.format(m=metric, mn=mean, s=s, N=N)


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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        #bi = '00000000_000001_00001'
        #bi = '00000000_000001_00008'
        #measure_all(blob_id=bi)
        #write_plate_timeseries_set(ex_id='00000000_000001')
        write_plate_percentiles(ex_id='00000000_000001', blob_ids=['00000000_000001_00001', 
                                                                   '00000000_000001_00002', 
                                                                   '00000000_000001_00003', 
                                                                   '00000000_000001_00004', 
                                                                   '00000000_000001_00005', 
                                                                   '00000000_000001_00006', 
                                                                   '00000000_000001_00007', 
                                                                   '00000000_000001_00008'])

    else:
        ex_ids = sys.argv[1:]
        for ex_id in ex_ids[:]:
            print 'searching for blobs for', ex_id
            measure_all_for_ex_id(ex_id)
