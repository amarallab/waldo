#!/usr/bin/env python
'''
Filename: plates.py
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
from annotation.experiment_index import organize_plate_metadata
from metrics.measurement_switchboard import pull_blob_data, quantiles_for_data
from metrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS
from wio.file_manager import get_good_blobs, get_dset
from wio.file_manager import write_table, read_table, df_equal

def write_plate_percentiles(ex_id, blob_ids=[], metrics=FULL_SET, autotest_write=True):
    if not blob_ids:
        blob_ids = get_good_blobs(ex_id)
        if not blob_ids:
            return
    
    plate_dataset = {}
    bad_blobs = []
    metric_index = []
    metrics_not_found = []

    for bID in blob_ids:
        blob_data = []
        blob_is_good = True
        metric_labels = []
        for metric in metrics:
            times, data = pull_blob_data(bID, metric=metric)
            if type(data) == None or len(data) == 0:
                metrics_not_found.append(metric)
                #print bID, metric, 'not found'
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

    print '{N} metrics | {a} blobs complete | {b} blobs incomplete'.format(N=len(metrics), a=len(blob_ids), b=len(bad_blobs))
    problem_metrics = sorted(list(set(metrics_not_found)))
    mlist=  ' | '.join(['{i} : {N}'.format(i=i, N=metrics_not_found.count(i)) for i in problem_metrics])
    print('missing types | {l}'.format(l=mlist))

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


    if autotest_write:
        p2 = read_table(ID=ex_id,
                        ID_type='plate',
                        data_type='percentiles',
                        dset=get_dset(ex_id),
                        file_tag='worm_percentiles')

        if df_equal(percentiles, p2):
            print 'write successful'
        else:
            print 'WARNING: WRITE NOT REPRODUCABLE'
            print 'before writing'
            print percentiles.head()
            print 'after reading'
            print p2.head()


def write_plate_timeseries(ex_id, blob_ids=[], data_types=FULL_SET[:], verbose=False, autotest_write=True):

    if blob_ids == None or len(blob_ids) == 0:
        blob_ids = get_good_blobs(ex_id)
    if blob_ids == None or len(blob_ids) == 0:
        return
            
    for data_type in data_types:
        print data_type, len(blob_ids) #, blob_ids[:4]
        blobs = []
        blob_ids = list(set(blob_ids))
        blob_series = []
        N = len(blob_ids[1:])
        for i, blob_id in enumerate(blob_ids[1:]):
            # calculate data_type for blob, skip if empty list returned        
            btimes, bdata = pull_blob_data(blob_id, metric=data_type)
            btimes = [round(t, ndigits=1) for t in btimes]

            if bdata == None:
                continue
            if len(bdata) == 0:
                continue
            blob_data = pd.DataFrame(bdata, columns=['data'])
            blob_data['time'] = btimes
            blob_data['bID'] = blob_id
            if verbose:
                print '\t{i} of {N} | {ID} | points: {p}'.format(i=i, N=N,
                                                                 ID=blob_id, p = len(blob_data))

            if len(blob_data):
                blob_series.append(blob_data)
            
        df = pd.concat(blob_series, axis=0)
        write_table(ID=ex_id,
                    ID_type='plate',
                    dataframe=df,
                    data_type=data_type,
                    dset=get_dset(ex_id),
                    file_tag='timeseries')

        if autotest_write:
            df2 = read_table(ID=ex_id,
                             ID_type='plate',
                             data_type=data_type,
                             dset=get_dset(ex_id),
                             file_tag='timeseries')

            if df_equal(df, df2):
                print 'write successful'
            else:
                print 'WARNING: WRITE NOT REPRODUCABLE'
                print 'before writing'
                print df.head()
                print 'after reading'
                print df2.head()

            
if __name__ == '__main__':
    dataset = 'disease_models'
    #data_type = 'cent_speed_bl'
    #data_type = 'length_mm'
    #data_type = 'curve_bl'
    eID = '20131211_145827'
    eID = '20130414_140704'

    write_plate_timeseries(ex_id=eID, data_types=['cent_speed_bl'])
    metrics = FULL_SET[:]
    write_plate_percentiles(ex_id=eID, metrics=metrics)
