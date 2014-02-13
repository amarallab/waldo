#!/usr/bin/env python

'''
Filename: temp_cache.py

Description:
'''

__author__ = 'Peter B. Winter'
__email__ = 'peterwinteriii@gmail.com'
__status__ = 'prototype'

#standard imports
import os
import sys
import json
from glob import glob
#from itertools import izip
 
# path definitions
CODE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
SHARED_DIR = CODE_DIR + 'shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from database.mongo_retrieve import pull_data_type_for_blob, timedict_to_list
from database.mongo_insert import insert_data_into_db, times_to_timedict
from settings.local import LOGISTICS, FILTER
from experiment_index import Experiment_Attribute_Index
from wio.blob_reader import Blob_Reader

TMP_DIR = os.path.dirname(os.path.realpath(__file__)) + '/tmp/'
        
def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def ensure_dir_present(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

ensure_dir_present(TMP_DIR)

def write_tmp_file(blob_id, data_type, data, tmp_dir=TMP_DIR):
    blob_path = '{path}/{bID}'.format(path=tmp_dir, bID=blob_id)
    ensure_dir_present(blob_path)
    tmp_file = '{path}/{dt}.json'.format(path=blob_path, dt=data_type)
    json.dump(data, open(tmp_file, 'w'))

def read_tmp_file(blob_id, data_type, tmp_dir=TMP_DIR):
    blob_path = '{path}/{bID}'.format(path=tmp_dir, bID=blob_id)
    ensure_dir_present(blob_path)
    if os.path.isdir(blob_path):
        tmp_file = '{path}/{dt}.json'.format(path=blob_path, dt=data_type)
        if os.path.isfile(tmp_file):
            return json.load(open(tmp_file, 'r'))
    return None
    
def clear_tmp_file(blob_id, data_type='all'):
    blob_path = '{path}/{bID}'.format(path=tmp_dir, bID=blob_id)
    tmp_file = '{path}/{dt}.json'.format(path=blob_path, dt=data_type)
    silent_remove(tmp_file)
    if data_type == 'all' or len(glob.glob(blob_path + '/*')) == 0:
        os.rmdir(blob_path)

def get_data(blob_id, data_type, split_time_and_data=True, **kwargs):
    tmp_data = read_tmp_file(blob_id=blob_id, data_type=data_type)
    # default: look in tmp file and split into 'time' and 'data'
    if tmp_data != None and split_time_and_data:
        times = tmp_data.get('time', [])
        data = tmp_data.get('data', [])
        if not times:
            print 'No Times Found! {dt} for {bi} not found'.format(dt=data_type, bi=blob_id)
        if not data:
            print 'No Data Found! {dt} for {bi} not found'.format(dt=data_type, bi=blob_id)
        return times, data, None
    # look in temp file but do not split results
    elif tmp_data != None:
        return tmp_data, None
    # temp file not located, attempt to find data in database.
    else:
        try:
            db_doc = pull_data_type_for_blob(blob_id, data_type, **kwargs)
            print 'warning: could not find tmp file for', blob_id, data_type
            # either split data into times and data or leave alone
            if split_time_and_data:
                times, data = timedict_to_list(db_doc['data'])
                return times, data, db_doc
            else:
                return db_doc['data'], db_doc

        # Stop everything if no to temp file or database doc.
        except Exception as e:
            print '\nFailure! could not locate data'
            print 'blob: {bID}\t type:{dt}\n'.format(bID=blob_id, dt=data_type)
            print e
            assert False
    return times, data, db_doc

def store_data_in_db(blob_id, data_type, data, description, db_doc=None, **kwargs):
    if not db_doc:
        db_doc = read_tmp_file(blob_id=blob_id, data_type='metadata')
    if not db_doc:
        try:
            db_doc = mongo_query({'blob_id':blob_id, 'data_type':'metadata'}, find_one=True)
            print 'warning: could not find tmp file for', blob_id, 'metadata'
        except Exception as e:
            print '\nFailure! could not locate data'
            print 'blob: {bID}\t type:{dt}\n'.format(bID=blob_id, dt='metadata')
            print e
            assert False
    insert_data_into_db(data, db_doc, data_type=data_type, 
                        description=description, **kwargs)                        
    return db_doc

'''                         

def store_data_in_db(blob_id, data_type, times, data, description, db_doc=None, **kwargs):
    # convert back to timedict form. then insert
    timedict = times_to_timedict(times, data)
    store_timedict_in_db(blob_id, data_type, timedict, description, db_doc, **kwargs)
'''    
        

        
    
