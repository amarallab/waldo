#!/usr/bin/env python

'''
Filename: consolidate_dataset.py
Description:

These functions are for consolidating plate summaries,
The plate summaries were created using consolidate_plates.py
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import datetime
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import glob
import scipy.stats as stats
import pandas as pd

# Path definitions
HERE = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.abspath(HERE + '/../')
PROJECT_HOME = os.path.abspath(CODE_DIR + '/../')
SHARED_DIR = CODE_DIR + '/shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

from wio.plate_utilities import get_plate_files, read_plate_timeseries, organize_plate_metadata, \
    return_flattened_plate_timeseries, write_dset_summary, format_dset_summary_name
from wio.file_manager import format_dirctory, ensure_dir_exists
from wormmetrics.measurement_switchboard import FULL_SET, STANDARD_MEASUREMENTS
from annotation.experiment_index import Experiment_Attribute_Index2

def get_annotations(dataset, data_type, label='all'):
    ex_ids, dfiles = get_plate_files(dataset=dataset, data_type=data_type)
    print len(ex_ids), 'ex_ids found for', dataset, data_type
    #print len(dfiles)
    ids, days, files = [], [], []
    labels = []
    for eID, dfile  in izip(ex_ids, dfiles):                
        #print eID, label
        hours, elabel, sub_label, pID, day = organize_plate_metadata(eID)
        if elabel not in labels:
            labels.append(elabel)
        if label == 'all' or str(label) == str(elabel):
            #print elabel, eID, day
            ids.append(eID)
            files.append(dfile)
            days.append(day)
    print labels
    return ids, days, files        

def generate_distribution(dataset, data_type, label, xlim, verbose=True):
    ex_ids, days, dfiles = get_annotations(dataset=dataset, data_type=data_type, label=label)
    print len(ex_ids), 'ex_ids found for', label
    #organize data by days
    #data_by_days = organize_plates_by_day(ex_ids, dfiles, days)
    data_by_days ={}
    for e, d, f in zip(ex_ids, days, dfiles):
        day = int(d[1:])
        if day not in data_by_days:
            data_by_days[day] = []
        data_by_days[day].append((e, f))
    if verbose:
        for d in sorted(data_by_days):
            print 'day', d, 'N:', len(data_by_days[d])

    # get a distribution for each day.
    #day_distributions = preprocess_distributions_by_day(data_by_days, xlim=xlim)
    day_distributions = {}
    day_quartiles = {}
    for day in sorted(data_by_days)[:]:
        print day
        all_data = []
        for eID, f in data_by_days[day][:]:
            plate_data = return_flattened_plate_timeseries(eID, dataset, data_type)
            all_data += list(plate_data)
            print len(all_data)
        s = all_data
        xmin, xmax = xlim
        bins = np.linspace(xmin, xmax, 5000)
        y, x = np.histogram(s, bins=bins)
        x = x[1:]
        y = np.array(y, dtype=float) / sum(y)
        day_distributions[day] = {'x':list(x), 'y':list(y)}
        day_quartiles[day] = [stats.scoreatpercentile(all_data, 25),
                                stats.scoreatpercentile(all_data, 50),
                                stats.scoreatpercentile(all_data, 75)]
    
    write_dset_summary(data=day_distributions, sum_type='dist', ID=label,
                       data_type=data_type, dataset=dataset)



def preprocess_distribution_set(dataset, labels=None, 
                                             data_types=STANDARD_MEASUREMENTS):
    xlims = {'cent_speed_bl':[0.0, 0.4], 
             'length_mm': [0.0, 2.0], 
             'curve_w': [0.0, 0.006], 
             'width_mm': [0.0, .2], 
             'angle_change': [0.0, 0.006]}

    if labels == None:
        ei = Experiment_Attribute_Index2(dataset)
        labels = [str(i) for i in set(ei['label'])]
        labels.append('all')

    for data_type in data_types:
        xlim = xlims.get(data_type, [0, 1])
        for label in labels:
            generate_distribution(dataset, data_type, label, xlim=xlim)


def combine_worm_percentiles_for_dset(dataset):    
    data_type = 'percentiles'
    tag = 'worm_percentiles'
    ex_ids, plate_files = get_plate_files(dataset=dataset,
                                          data_type=data_type,
                                          tag=tag)

    all_blob_ids, all_percentiles = [], None
    for ex_id in ex_ids:
        #ex_id = pf.split('/')[-1].split('-')[0]
        #print ex_id
        blob_ids, percentiles = read_plate_timeseries(ex_id,
                                                      dataset=dataset,
                                                      data_type=data_type,
                                                      tag=tag)
        if all_percentiles == None:
            all_blob_ids = list(blob_ids)
            all_percentiles = percentiles
        else:
            all_blob_ids += list(blob_ids)
            all_percentiles = np.concatenate((all_percentiles, percentiles))
        #print len(all_blob_ids), all_percentiles.shape

    nan_count = 0
    data_types = []
    for i, row in enumerate(all_percentiles):
        for i in row:
            data_types.append(type(i))
            if np.isnan(i):
                nan_count += 1
                print 'nan in row:', i

    print 'total NaNs found:', nan_count
    print 'types found:', list(set(data_types))
    return all_blob_ids, all_percentiles

# probably not used.
'''
def create_full_worm_index(blob_ids):
    ex_id_data, worm_rows = {}, []    
    for blob_id in blob_ids:
        ex_id = '_'.join(blob_id.split('_')[:2])
        #print ex_id
        if ex_id not in ex_id_data:
            ex_id_data[ex_id] = organize_plate_metadata(ex_id)
        worm_rows.append(ex_id_data[ex_id])
    return worm_rows
'''
                    
def consolidate_dset_from_plate_timeseries(dataset, data_type, verbose=True):    
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

def write_combined_worm_percentiles(dataset):
    # manage paths for files
    save_dir = format_dirctory(ID=dataset, ID_type='dset')
    f_savename = '{path}{dset}_features.csv'.format(path=save_dir, dset=dataset)
    i_savename = '{path}{dset}_index.csv'.format(path=save_dir, dset=dataset)    
    ensure_dir_exists(save_dir)
    # combine worm percentiles and indexes for worms
    blob_ids, percentiles = combine_worm_percentiles_for_dset(dataset)
    #worm_index = create_full_worm_index(blob_ids)
    ex_id_data, worm_index = {}, []    
    for blob_id in blob_ids:
        ex_id = '_'.join(blob_id.split('_')[:2])
        #print ex_id
        if ex_id not in ex_id_data:
            ex_id_data[ex_id] = organize_plate_metadata(ex_id)
        worm_index.append(ex_id_data[ex_id])
    print '{N} blob ids included'.format(N=len(blob_ids))
    # write features
    pf = pd.DataFrame(percentiles, index=blob_ids)
    pf.to_csv(f_savename, header=False)
    # write worm index             
    wi = pd.DataFrame(worm_index, index=blob_ids)
    wi.to_csv(i_savename, header=False)    

    for line in worm_index:
        print line

def write_dset_summaries(dataset, measurements=STANDARD_MEASUREMENTS):
    for data_type in measurements:
        print 'consolidating data for {dt}'.format(dt=data_type)
        data = consolidate_dset_from_plate_timeseries(dataset, data_type)
        print dataset, data_type, data
        write_dset_summary(data=data, sum_type='basic', 
                           data_type=data_type, dataset=dataset)

if __name__ == '__main__':
    dataset = 'disease_models'
    #dataset = 'thermo_recovery'
    #dataset = 'copas_TJ3001_lifespan'
    #write_combined_worm_percentiles(dataset)
    #write_dset_summaries(dataset)
    #generate_distribution(dataset)
    preprocess_distribution_set(dataset)