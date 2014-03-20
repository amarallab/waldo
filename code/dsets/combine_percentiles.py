#!/usr/bin/env python
'''
Filename: compile_plates_for_clustering.py
Description:


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

from wio.plate_utilities import get_plate_files, read_plate_timeseries, organize_plate_metadata
from wio.file_manager import format_dirctory, ensure_dir_exists

def get_combined_worm_percentiles(dataset):
    
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

def create_full_worm_index(blob_ids):
    ex_id_data, worm_rows = {}, []    
    for blob_id in blob_ids:
        ex_id = '_'.join(blob_id.split('_')[:2])
        #print ex_id
        if ex_id not in ex_id_data:
            ex_id_data[ex_id] = organize_plate_metadata(ex_id)
        worm_rows.append(ex_id_data[ex_id])
    return worm_rows
        
        
def combine_worm_percentiles(dataset):
    # manage paths for files
    save_dir = format_dirctory(ID=dataset, ID_type='dset')
    f_savename = '{path}{dset}_features.csv'.format(path=save_dir, dset=dataset)
    i_savename = '{path}{dset}_index.csv'.format(path=save_dir, dset=dataset)    
    ensure_dir_exists(save_dir)
    # combine worm percentiles and indexes for worms
    blob_ids, percentiles = get_combined_worm_percentiles(dataset)
    worm_index = create_full_worm_index(blob_ids)
    print '{N} blob ids included'.format(N=len(blob_ids))
    # write features
    pf = pd.DataFrame(percentiles, index=blob_ids)
    pf.to_csv(f_savename, header=False)
    # write worm index             
    wi = pd.DataFrame(worm_index, index=blob_ids)
    wi.to_csv(i_savename, header=False)    

    for line in worm_index:
        print line
    
if __name__ == '__main__':
    #dataset = 'disease_models'
    dataset = 'thermo_recovery'
    #dataset = 'copas_TJ3001_lifespan'
    blob_ids = combine_worm_percentiles(dataset)

    
