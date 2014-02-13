#!/usr/bin/env python
'''
Filename: worm_timeseries.py
Description: Pulls data from database and writes multiple time series for each blob into json.
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
import json

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

# nonstandard imports
from database.mongo_retrieve import unique_blob_ids_for_query
from WormMetrics.spine_measures import spine_measures

def data_to_json(query, savedir='.'):
    blob_ids = unique_blob_ids_for_query(query)
    print len(blob_ids), 'blob_ids found'
    for blob_id in blob_ids:
        print blob_id
        timeseries_dict = spine_measures(blob_id, for_plotting=True)
        try:
            timeseries_dict = spine_measures(blob_id, for_plotting=True)
            savename = savedir + '/' + blob_id + '_spine_measures.json'
            print 'writing:', savename
            json.dump(timeseries_dict, open(savename, 'w'))
        except Exception as e:
            print e

if __name__ == '__main__':
    query = {'age':'A3', 'data_type':'metadata'}
    data_to_json(query)
