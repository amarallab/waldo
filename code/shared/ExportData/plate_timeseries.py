#!/usr/bin/env python
'''
Filename: plate_timeseries.py
Description: aggregates all measured values for all worms from a single recording.

'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

# standard imports
import os
import sys
from itertools import izip
import json

# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../../'
SHARED_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(SHARED_DIR)
sys.path.append(CODE_DIR)

# nonstandard imports
from WormMetrics.switchboard import pull_all_for_ex_id, pull_metric_for_blob_id
from database.mongo_retrieve import mongo_query, timedict_to_list
import database.mongo_support_functions as mongo
from settings.local import LOGISTICS, MONGO

#Globals
DATA_DIR = LOGISTICS['filesystem_data']
DEFAULT_DIR = LOGISTICS['export']

'''
def get_all_timepoints_for_ex_id(ex_id, data_dir=DATA_DIR):
    search_path = '{path}/{ex_id}/*summary'.format(path=data_dir, ex_id=ex_id)
    s_files = glob.glob(search_path)
    if len(s_files) != 1:
        print 'Warning: {N} summary files found for {eID}'.format(N=len(s_files), eID=ex_id)
    with open(s_files[0], 'r') as f:
        times = [line.split()[1] for line in f]
    return times
'''

def write_full_plate_timeseries(ex_id, metric='centroid_speed', savename='test.txt', as_json=False, **kwargs):

    # make list of all blobs
    stage1_data = mongo_query({'ex_id': ex_id, 'type': 'stage1'}, find_one=True, col='plate_collection', **kwargs)
    blob_ids = None
    if stage1_data:
        blob_ids = stage1_data.get('blob_ids', None)

    if not blob_ids:
        blob_ids = [e['blob_id'] for e in mongo_query({'ex_id': ex_id, 'data_type':'smoothed_spine'}, {'blob_id':1}, **kwargs)]
        blob_ids = list(set(blob_ids))
        print 'plate document failed, pulling directly from blob collection', len(blob_ids), 'blob_ids found'

    # make giant dict of all values.
    data_dict = {}
    data_timedict = {}
    for blob_id in blob_ids:
        data_timedict = pull_metric_for_blob_id(blob_id=blob_id, metric=metric, remove_skips=True, **kwargs)
        if len(data_timedict) == 0:
            print blob_id, 'has not data points, apparently'
        for key, value in data_timedict.iteritems():
            new_key = ('%.1f' % float(key.replace('?', '.'))).replace('.', '?')
            if new_key not in data_dict:
                data_dict[new_key] = []
            data_dict[new_key].append(value)

    # if empty, write anyway.
    times, data = [], []
    if len(data_timedict) == 0:
        print ex_id, 'has no data'

    if as_json:
        json.dump(data_timedict, savename)
    else:
        if len(data_timedict) > 1:
            times, data = timedict_to_list(data_dict)
        with open(savename, 'w') as f:
            for t, d in izip(times, data):
                line = '{t},{l}'.format(t=t, l=','.join(map(str, d)))
                f.write(line + '\n')


def main(**kwargs):
    ex_id = '20130320_102312'
    write_full_plate_timeseries(ex_id, **kwargs)



if __name__ == '__main__':
    mongo_client, _ = mongo.start_mongo_client(MONGO['ip'], MONGO['port'],
                                               MONGO['database'], MONGO['blobs'])
    try:

        main(mongo_client=mongo_client)
    finally:
        mongo.close_mongo_client(mongo_client=mongo_client)
