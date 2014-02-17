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
HERE = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(HERE + '/../../')
CODE_DIR = os.path.abspath(HERE + '/../')
SHARED_DIR = CODE_DIR + 'shared'
sys.path.append(CODE_DIR)
sys.path.append(SHARED_DIR)

# nonstandard imports
from database.mongo_retrieve import pull_data_type_for_blob, timedict_to_list
from database.mongo_insert import insert_data_into_db, times_to_timedict
from settings.local import LOGISTICS
#from annotation.experiment_index import Experiment_Attribute_Index


from wio.blob_reader import Blob_Reader

EXPORT_PATH = LOGISTICS['export']
TMP_DIR = PROJECT_HOME + '/data/processing/'
        
def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
   
def ensure_dir_exists(path):
    ''' recursivly creates path in filesystem, if it does not exist '''
    path = os.path.abspath(path)
    savedir = ''
    for i, d in enumerate(path.split('/')):
        if d:
            savedir += '/{d}'.format(d=d)        
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
                print 'created:{d}'.format(d=savedir)
    return savedir

def manage_save_path(out_dir, path_tag, ID, data_type):
    ''' returns a unique descriptive file name to store data and makes sure path to it exists'''
    # get directories in order
    out_dir = '{d}/{b}/'.format(d=out_dir.rstrip('/'), b=path_tag.lstrip('/'))
    out_dir = ensure_dir_exists(out_dir.rstrip('/'))

    now_string = time.ctime().replace('  ', '_').replace(' ', '_')
    now_string = now_string.replace(':', '.').strip()
    save_name = '{path}/{ID}_{dt}_time-{n}.json'.format(path=out_dir, ID=ID, 
                                                        dt=data_type, n=now_string)
    print save_name
    return save_name


def get_ex_ids(query, **kwargs):
    ''' return a list of unique ex_id names for a query'''
    return list(set([e['ex_id'] for e in mongo_query(query=query, projection={'ex_id':1}, **kwargs)]))

def get_blob_ids(query, **kwargs):
    ''' return a list of unique blob_id names for a query'''
    return list(set([e['blob_id'] for e in mongo_query(query=query, projection={'blob_id':1}, **kwargs)]))
    
def format_tmp_filename(blob_id, data_type, tmp_dir):
    ex_id = '_'.join(blob_id.split('_')[:2])
    blob_path = '{path}/{eID}'.format(path=tmp_dir, eID=ex_id)
    ensure_dir_exists(blob_path)
    tmp_file = '{path}/{bID}-{dt}.json'.format(path=blob_path, bID=blob_id,
                                               dt=data_type)
    return tmp_file

def write_tmp_file(blob_id, data_type, data, tmp_dir=TMP_DIR):
    tmp_file = format_tmp_filename(blob_id, data_type, tmp_dir=tmp_dir)
    json.dump(data, open(tmp_file, 'w'))

def read_tmp_file(blob_id, data_type, tmp_dir=TMP_DIR):
    tmp_file = format_tmp_filename(blob_id, data_type, tmp_dir=tmp_dir)
    if os.path.isfile(tmp_file):
        return json.load(open(tmp_file, 'r'))    
    '''
    ex_id = '_'.join(blob_id.split('_')[:1])    
    blob_path = '{path}/{eID}'.format(path=tmp_dir, eID=ex_id)
    ensure_dir_exists(blob_path)
    if os.path.isdir(blob_path):
        tmp_file = '{path}/{bID}-{dt}.json'.format(path=blob_path, dt=data_type)
        if os.path.isfile(tmp_file):
            return json.load(open(tmp_file, 'r'))
    '''
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
       
ensure_dir_exists(TMP_DIR)
        
    
