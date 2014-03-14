#!/usr/bin/env python

'''
Filename: get_plate_timeseries.py
Description:
Pull one type of data out of database, and save it in jsons organized by ex_id.
data pulled is broken into 15 minute segments. within each 15min segment data is pulled either
by subsampling the data or by binning into 10 second bins.
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

# Path definitions
project_directory = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(project_directory)

# nonstandard imports
from Shared.Code.Database.mongo_retrieve import mongo_query, pull_data_type_for_blob, timedict_to_list
from Import.Code.experiment_index import Experiment_Attribute_Index
from Shared.Code.WormMetrics.switchboard import pull_metric_for_blob_id
from Shared.Code.Settings.data_settings import mongo_settings
import Shared.Code.Database.mongo_support_functions as mongo
from Shared.Code.ExportData.plate_timeseries import write_full_plate_timeseries
from create_plate_document import create_stage1_plate_doc

# Globals
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Data/Time-Series/'

def choose_ex_id(key='purpose', value='N2_aging', **kwargs):
    ei = Experiment_Attribute_Index()
    return ei.return_ex_ids_with_attribute(key_attribute=key, attribute_value=value, **kwargs)

def find_ex_ids_present():
    search_path = '{path}*_*.txt'.format(path=DATA_DIR)
    data_files = glob.glob(search_path)
    ex_ids_present = []
    for f in data_files:
        ex_id = f.split('/')[-1].split('.txt')[0]
        ex_ids_present.append(ex_id)
    return ex_ids_present

def main(**kwargs):
    '''
    ex_id = '20130320_102312'
    savename = '{path}{ex_id}.txt'.format(path=Data_DIR, ex_id=ex_id)
    write_full_plate_timeseries(ex_id, savename=savename, **kwargs)
    '''
    purpose='N2_aging'
    data_type='curvature_all_bl'
    all_ex_ids = choose_ex_id(key='purpose', value=purpose, **kwargs)
    ex_ids_present = find_ex_ids_present()
    print len(all_ex_ids), len(ex_ids_present)
    ex_ids = list(set(all_ex_ids) - set(ex_ids_present))
    print len(ex_ids)

    for ex_id in ex_ids[:]:
        #create_stage1_plate_doc(ex_id, **kwargs)
        print ex_id        
        out_dir = '{path}{p}-{dt}'.format(path=DATA_DIR, p=purpose, dt=data_type)
        print out_dir
        write_full_plate_timeseries(ex_id, metric=data_type, out_dir=out_dir, as_json=True, **kwargs)

        '''
        try:
            savename = '{path}{ex_id}.txt'.format(path=DATA_DIR, ex_id=ex_id)
            write_full_plate_timeseries(ex_id, savename=savename, **kwargs)
        except Exception as e:
            print ex_id, e
        '''

if __name__ == '__main__':
    mongo_client, _ = mongo.start_mongo_client(mongo_settings['mongo_ip'], mongo_settings['mongo_port'],
                                               mongo_settings['worm_db'], mongo_settings['blob_collection'])
    try:

        main(mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)
